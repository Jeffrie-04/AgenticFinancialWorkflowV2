"""
afw/ingest.py — validate a bank CSV against the transaction contract (NO LLM).

Runs before any model sees the data. Every non-blank row comes out as a
Transaction with a status (OK / NEEDS_REVIEW / REJECTED) and a reason; if more
than 10% of rows are REJECTED the whole file fails, before any LLM spend.

Parsed with the stdlib csv module, not pandas: pandas would coerce amounts to
float (hiding "19.999" precision errors) and lose physical line numbers.

INPUT : <business>/transactions.csv   columns: date, merchant, amount,
        description (optional), currency (optional)
OUTPUT: outputs/ingested.json      validated transactions
        outputs/data_quality.json  counts by status/reason, $ needing review
"""

import csv
import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation

from afw.models import (
    DESCRIPTION_MAX,
    MAX_AMOUNT,
    Direction,
    SignConvention,
    SourceConfig,
    Status,
    Transaction,
    date_format_regex,
    normalize_merchant,
)

REJECT_THRESHOLD = 0.10  # file fails when REJECTED / rows is strictly above this
REQUIRED_COLUMNS = {"date", "merchant", "amount"}
# Plain decimal only; "$1,234.56", "(12.00)", "1E+3" etc. are out of scope (Phase 1).
AMOUNT_RE = re.compile(r"[+-]?\d+(\.\d+)?")
NON_USD_SYMBOLS = re.compile(r"[€£¥₹]")
# "return" deliberately excluded: too noisy ("Tax return prep").
REFUND_KEYWORDS = re.compile(r"\b(refund|reversal|chargeback)\b", re.IGNORECASE)
CENT = Decimal("0.01")

# Per-source config, keyed by business folder name. Anything else (e.g. an
# upload in businesses/_uploads/<uuid>/) gets the default SourceConfig().
SOURCE_CONFIGS = {
    "landscaper": SourceConfig(),
    "law_firm": SourceConfig(),
    "restaurant": SourceConfig(),
}


class IngestFailed(Exception):
    """The file as a whole is unusable (too many rejects, no rows, bad header)."""


@dataclass
class IngestResult:
    source_file: str
    rows: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    description_truncated: int = 0

    @property
    def counts(self):
        by_status = Counter(t.status for t in self.rows)
        return {
            "rows": len(self.rows),
            "ok": by_status[Status.OK],
            "needs_review": by_status[Status.NEEDS_REVIEW],
            "rejected": by_status[Status.REJECTED],
        }


def config_for(csv_path):
    name = os.path.basename(os.path.dirname(os.path.abspath(csv_path)))
    return SOURCE_CONFIGS.get(name, SourceConfig())


def read_rows(path):
    """-> ([(line_no, {column: value} or None)], warnings) for non-blank rows.
    None marks a row whose field count doesn't match the header (e.g. an
    unquoted "1,234.56"), which can't be mapped to columns safely.
    UTF-8 (BOM stripped) first; cp1252 fallback for Excel-on-Windows exports."""
    warnings = []
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("cp1252")
        except UnicodeDecodeError as e:
            raise IngestFailed(f"{os.path.basename(path)}: not UTF-8 or cp1252 ({e})")
        warnings.append("encoding_fallback_cp1252")

    # strict: a malformed file (e.g. an unterminated quote, which would
    # otherwise swallow the following rows into one field) fails outright.
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        return _read_records(reader, os.path.basename(path)), warnings
    except csv.Error as e:
        raise IngestFailed(
            f"{os.path.basename(path)}: malformed CSV at line {reader.line_num}: {e}") from e


def _read_records(reader, source_file):
    header = [h.strip().lower() for h in next(reader, [])]
    duplicates = sorted(h for h, n in Counter(header).items() if n > 1)
    if duplicates:
        raise IngestFailed(f"{source_file}: duplicate columns {duplicates}")
    missing = REQUIRED_COLUMNS - set(header)
    if missing:
        raise IngestFailed(f"{source_file}: missing columns {sorted(missing)}")

    rows = []
    for values in reader:
        if not any(v.strip() for v in values):  # empty line or ",,,,"
            continue
        mapped = dict(zip(header, values)) if len(values) == len(header) else None
        rows.append((reader.line_num, mapped))
    return rows


def parse_amount(raw, cfg):
    """-> (amount > 0 with 2dp, direction, None) or (None, None, reason)."""
    s = raw.strip()
    if not s:
        return None, None, "amount_blank"
    if NON_USD_SYMBOLS.search(s):
        return None, None, "non_usd"
    if not AMOUNT_RE.fullmatch(s):
        return None, None, "amount_unparseable"
    try:
        value = Decimal(s)
    except InvalidOperation:
        return None, None, "amount_unparseable"
    if value == 0:
        return None, None, "amount_zero"
    if abs(value) > MAX_AMOUNT:  # also keeps quantize() inside Decimal's precision
        return None, None, "amount_out_of_range"
    if value.quantize(CENT) != value:
        return None, None, "amount_precision"

    negative_is_credit = cfg.sign_convention == SignConvention.NEGATIVE_IS_CREDIT
    direction = Direction.CREDIT if (value < 0) == negative_is_credit else Direction.DEBIT
    return abs(value).quantize(CENT), direction, None


def parse_date(raw, cfg):
    """-> (date, None) or (None, reason). Wrong layout vs impossible date are
    reported separately so a misconfigured source is easy to spot."""
    s = raw.strip()
    if not date_format_regex(cfg.date_format).fullmatch(s):
        return None, "date_format"
    try:
        return datetime.strptime(s, cfg.date_format).date(), None
    except ValueError:
        return None, "date_invalid"


def parse_row(raw, line_no, source_file, cfg):
    """One CSV row -> (Transaction fields, description_was_truncated). Checks
    run amount, currency, date, merchant; the first failure is the reason.
    The id is assigned later, once duplicates in the file are known."""
    fields = {
        "source_file": source_file,
        "source_row": line_no,
        "account_id": cfg.account_id,
        "status": Status.OK,
    }
    if raw is None:
        return {**fields, "status": Status.REJECTED, "reason": "row_field_count"}, False

    description = raw.get("description", "").strip()
    truncated = len(description) > DESCRIPTION_MAX
    fields.update(merchant=raw["merchant"].strip(), description=description[:DESCRIPTION_MAX])

    def reject(reason):
        return {**fields, "status": Status.REJECTED, "reason": reason}, truncated

    amount, direction, reason = parse_amount(raw.get("amount", ""), cfg)
    if reason:
        return reject(reason)
    currency = raw.get("currency", "").strip().upper() or cfg.currency
    if currency != "USD":
        return reject("non_usd")
    row_date, reason = parse_date(raw.get("date", ""), cfg)
    if reason:
        return reject(reason)

    fields.update(date=row_date, amount=amount, direction=direction, currency=currency)
    if not fields["merchant"]:
        # Contract: blank merchant is NEEDS_REVIEW unless a clear transfer;
        # transfer detection is out of scope for Phase 1.
        fields.update(status=Status.NEEDS_REVIEW, reason="merchant_blank")
    return fields, truncated


def assign_ids(parsed):
    """id = content hash + occurrence number, so two identical rows (two $4.50
    coffees on the same day) stay distinct and both are kept."""
    seen = Counter()
    for f in parsed:
        key = "|".join(str(f.get(k) or "") for k in
                       ("account_id", "date", "amount", "direction", "merchant", "description"))
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        f["id"] = f"{digest}-{seen[digest]}"
        seen[digest] += 1


def detect_refunds(parsed):
    """A CREDIT is a refund when the same merchant (normalize_merchant) has an
    OK DEBIT on the same day or earlier with enough unrefunded balance left;
    refunds use up the oldest such DEBIT first. A CREDIT that only matches the
    merchant, or only mentions a refund keyword, is NEEDS_REVIEW. Outcomes
    never depend on row order within a day."""
    usable = [f for f in parsed if f["status"] == Status.OK]
    merchant_key = lambda f: normalize_merchant(f["merchant"])
    merchants_with_debits = {merchant_key(f) for f in usable if f["direction"] == Direction.DEBIT}
    balances = defaultdict(list)  # merchant -> [[debit fields, remaining], ...] oldest first
    canonical = lambda f: (-f["amount"], f["id"])  # row-order-free tiebreak within a day

    days = defaultdict(lambda: ([], defaultdict(list)))  # date -> (debits, merchant -> credits)
    for f in usable:
        debits, credits = days[f["date"]]
        if f["direction"] == Direction.DEBIT:
            debits.append(f)
        else:
            credits[merchant_key(f)].append(f)

    for day in sorted(days):
        debits, credits = days[day]
        # Same-day DEBITs are added before that day's CREDITs: "same day counts".
        for f in sorted(debits, key=canonical):
            balances[merchant_key(f)].append([f, f["amount"]])
        for merchant, group in credits.items():
            available = sum(b[1] for b in balances[merchant])
            if len(group) > 1 and 0 < available < sum(f["amount"] for f in group):
                # Several same-day credits competing for one balance: which one
                # is the refund can't be told from the data.
                for f in group:
                    f.update(status=Status.NEEDS_REVIEW, reason="refund_ambiguous")
                continue
            for f in sorted(group, key=canonical):
                match = next((b for b in balances[merchant] if b[1] >= f["amount"]), None)
                if match:
                    match[1] -= f["amount"]
                    f.update(is_refund=True, refund_of=match[0]["id"])
                elif merchant in merchants_with_debits:
                    f.update(status=Status.NEEDS_REVIEW, reason="refund_merchant_match_only")
                elif REFUND_KEYWORDS.search(f"{f['merchant']} {f['description']}"):
                    f.update(status=Status.NEEDS_REVIEW, reason="refund_keyword_only")


def ingest_rows(path, cfg):
    """Parse and validate every row. Never raises for bad rows."""
    source_file = os.path.basename(path)
    raw_rows, warnings = read_rows(path)
    result = IngestResult(source_file=source_file, warnings=warnings)

    parsed = []
    for line_no, raw in raw_rows:
        fields, truncated = parse_row(raw, line_no, source_file, cfg)
        parsed.append(fields)
        result.description_truncated += truncated
    assign_ids(parsed)
    detect_refunds(parsed)

    result.rows = [Transaction(**f) for f in parsed]
    return result


def ingest_file(path, cfg):
    """ingest_rows + the file-level gate: fail if > 10% of rows are REJECTED."""
    result = ingest_rows(path, cfg)
    counts = result.counts
    if counts["rows"] == 0:
        raise IngestFailed(f"{result.source_file}: no data rows")
    rate = counts["rejected"] / counts["rows"]
    if rate > REJECT_THRESHOLD:
        top = Counter(t.reason for t in result.rows if t.status == Status.REJECTED).most_common(3)
        raise IngestFailed(
            f"{result.source_file}: {counts['rejected']}/{counts['rows']} rows rejected "
            f"({rate:.1%} > {REJECT_THRESHOLD:.0%}); top reasons: {dict(top)}")
    return result


def data_quality(result):
    review = [t for t in result.rows if t.status == Status.NEEDS_REVIEW]
    total = lambda d: str(sum((t.amount for t in review if t.direction == d), Decimal("0.00")))
    return {
        "source_file": result.source_file,
        "counts": result.counts,
        "reasons": dict(Counter(t.reason for t in result.rows if t.reason)),
        "needs_review_total": {
            "debit": total(Direction.DEBIT),
            "credit": total(Direction.CREDIT),
            "rows": len(review),
        },
        "description_truncated": result.description_truncated,
        "warnings": result.warnings,
    }


def main(csv_path="data/transactiondata.csv", outputs_dir="outputs"):
    result = ingest_file(csv_path, config_for(csv_path))

    ingested_path = os.path.join(outputs_dir, "ingested.json")
    with open(ingested_path, "w") as f:
        json.dump({
            "source_file": result.source_file,
            "transactions": [t.model_dump(mode="json") for t in result.rows],
        }, f, indent=2)

    dq_path = os.path.join(outputs_dir, "data_quality.json")
    with open(dq_path, "w") as f:
        json.dump(data_quality(result), f, indent=2)

    c = result.counts
    print(f"Ingested {c['rows']} rows: {c['ok']} OK, {c['needs_review']} NEEDS_REVIEW, "
          f"{c['rejected']} REJECTED -> {ingested_path}")
    return result


if __name__ == "__main__":
    main()
