"""
phase3_kpisnoAI.py — Deterministic KPI engine (NO LLM).

DESIGN DECISION (the headline of this project):
Every metric in this file is exact arithmetic on the transaction data, so it
is computed in plain Python and NEVER sent to a language model. The LLM in
this pipeline handles fuzzy work (categorization, summaries); money math has
to be correct and reproducible, so it lives here. That separation is
deliberate — an LLM's arithmetic can't be trusted for financial figures.

INPUT : outputs/ingested.json     (validated source rows from afw/ingest.py)
        Amount, direction, date, merchant and status come ONLY from here.
        outputs/categorized.json  (produced by the LLM categorizer)
        Only the category is taken from here, joined on (date, merchant,
        |amount|); the LLM's echoed amounts and signs are never used.
OUTPUT: outputs/kpis.json
        outputs/data_quality.json  ("kpi_join" section added)
"""

import json
import os
from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from afw.models import Category, Direction, Status, normalize_merchant

# Categories treated as committed/fixed monthly obligations vs discretionary.
# NOTE: this is a category-based heuristic, not true recurrence detection.
# Detecting real month-over-month recurrence needs multi-month data, which a
# single statement doesn't give. Utilities here holds SaaS, rent, payroll,
# insurance, and bills — costs owed regardless of revenue.
# (.value: rows carry plain strings, and a str-Enum member hashes by name.)
FIXED_CATEGORIES = {Category.UTILITIES.value}
DISCRETIONARY_CATEGORIES = {Category.DINING.value, Category.SHOPPING.value, Category.OTHER.value}

# Reasons a source row that passed ingest is still left out of the KPIs.
JOIN_PROBLEMS = ("join_ambiguous", "join_missing", "category_invalid", "refund_original_excluded")


def parse_date(raw):
    """Parse a statement date. Primary format MM-DD-YYYY; falls back to ISO.
    Returns a datetime (or the date itself if already one), or None if
    unparseable (so bad rows are flagged, not silently dropped)."""
    if isinstance(raw, date):
        return raw
    for fmt in ("%m-%d-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(raw), fmt)
        except ValueError:
            continue
    return None


def load_transactions(path="outputs/categorized.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return data["categorized"]


def load_ingested(path="outputs/ingested.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return data["transactions"]


def _categorized_key(t):
    """(date, merchant key, |amount|) for an LLM-echoed row, or None if the
    LLM mangled it. The LLM may reformat dates or flip signs; neither matters."""
    d = parse_date(t.get("date"))
    try:
        amount = abs(Decimal(str(t["amount"])))
    except (InvalidOperation, KeyError):
        return None
    if d is None or not amount.is_finite():
        return None
    d = d.date() if isinstance(d, datetime) else d
    return d, normalize_merchant(str(t.get("merchant", ""))), amount


def join_categories(ingested, categorized):
    """Attach the LLM's category to each OK source row. Returns (rows, report).

    A row is left out of the KPIs (NEEDS_REVIEW, counted, never silently
    dropped) when its key appears more than once in the source, the LLM rows
    for it disagree, no LLM row matches, or the category isn't a Category.
    A refund takes no category of its own — it is netted against its
    original DEBIT — so it is left out only if that DEBIT is.
    """
    # TODO(phase 2): categorizer returns {id, category} only; replace this key
    # join with an id lookup.
    report = Counter()
    by_key = defaultdict(list)
    for c in categorized:
        key = _categorized_key(c)
        if key is None:
            report["categorized_invalid"] += 1
        else:
            by_key[key].append(c)

    usable = []
    for t in ingested:
        if t["status"] == Status.REJECTED.value:
            continue
        row = {**t, "date": date.fromisoformat(t["date"])}
        usable.append((row, (row["date"], normalize_merchant(row["merchant"]), Decimal(row["amount"]))))
    source_keys = Counter(key for _, key in usable)
    report["categorized_unmatched"] = sum(len(v) for k, v in by_key.items() if k not in source_keys)

    excluded = []

    def exclude(row, reason):
        excluded.append({**row, "status": Status.NEEDS_REVIEW.value, "reason": reason})
        report[reason] += 1

    joined = []
    for row, key in usable:
        if row["status"] != Status.OK.value:
            continue  # ingest already marked it NEEDS_REVIEW and counted it
        if row["is_refund"]:
            joined.append(row)
            continue
        matches = by_key.get(key, [])
        categories = {str(m.get("category")) for m in matches}
        if source_keys[key] > 1 or len(categories) > 1:
            exclude(row, "join_ambiguous")
        elif not matches:
            exclude(row, "join_missing")
        elif categories.pop() not in {c.value for c in Category}:
            exclude(row, "category_invalid")
        else:
            report["categorized_duplicate"] += len(matches) - 1
            joined.append({**row, "category": matches[0]["category"]})

    kept_debits = {r["id"] for r in joined if not r["is_refund"]}
    rows = []
    for row in joined:
        if row["is_refund"] and row["refund_of"] not in kept_debits:
            exclude(row, "refund_original_excluded")
        else:
            rows.append(row)

    total = lambda d: str(sum((Decimal(r["amount"]) for r in excluded if r["direction"] == d),
                              Decimal("0.00")))
    return rows, {
        "rows_in_kpis": len(rows),
        **{k: report[k] for k in JOIN_PROBLEMS},
        "categorized_unmatched": report["categorized_unmatched"],
        "categorized_duplicate": report["categorized_duplicate"],
        "categorized_invalid": report["categorized_invalid"],
        "excluded_total": {
            "debit": total(Direction.DEBIT.value),
            "credit": total(Direction.CREDIT.value),
            "rows": len(excluded),
        },
    }


def apply_refunds(transactions):
    """Net each refund against the DEBIT it refunds, so a returned purchase
    lowers that DEBIT's spend (same category, same merchant) instead of
    counting as income. Refund rows stay in the list; split_income_expense
    skips them."""
    refunded = defaultdict(float)
    for t in transactions:
        if t.get("is_refund"):
            refunded[t["refund_of"]] += float(t["amount"])
    return [{**t, "amount": float(t["amount"]) - refunded[t["id"]]} if t.get("id") in refunded else t
            for t in transactions]


def split_income_expense(transactions):
    """CREDIT is income, DEBIT is an expense. Refund CREDITs are neither:
    apply_refunds() has already netted them against their DEBIT."""
    income, expense = [], []
    for t in transactions:
        if t.get("is_refund"):
            continue
        (income if t["direction"] == Direction.CREDIT.value else expense).append(t)
    return income, expense


def compute_core_kpis(transactions):
    """The original four KPIs, preserved."""
    income, expense = split_income_expense(transactions)

    # total_spend = sum of all positive amounts
    total_spend = sum(float(t["amount"]) for t in expense)
    # total_income = sum of the absolute value of negative amounts
    total_income = sum(abs(float(t["amount"])) for t in income)

    # top_merchants = 3 merchants with the highest total spend
    merchant_spend = defaultdict(float)
    for t in expense:
        merchant_spend[t["merchant"]] += float(t["amount"])
    top_merchants = [m for m, _ in sorted(
        merchant_spend.items(), key=lambda x: x[1], reverse=True)[:3]]

    # average_expense = total_spend / number of expense transactions
    average_expense = total_spend / len(expense) if expense else 0.0

    return {
        "total_spend": round(total_spend, 2),
        "total_income": round(total_income, 2),
        "top_merchants": top_merchants,
        "average_expense": round(average_expense, 2),
    }


def compute_net_cash_flow(transactions):
    """net_cash_flow = total_income - total_spend.
    The single most important 'am I okay' number: did more money come in than
    went out this period. Positive = surplus, negative = burning cash."""
    income, expense = split_income_expense(transactions)
    total_income = sum(abs(float(t["amount"])) for t in income)
    total_spend = sum(float(t["amount"]) for t in expense)
    net = total_income - total_spend
    return {
        "net_cash_flow": round(net, 2),
        "status": "surplus" if net >= 0 else "deficit",
    }


def compute_spend_by_category(transactions):
    """spend per category, and each as a % of total spend.
    pct = (category_spend / total_spend) * 100.
    This is where an owner finds costs to cut."""
    _, expense = split_income_expense(transactions)
    total_spend = sum(float(t["amount"]) for t in expense)

    cat_spend = defaultdict(float)
    for t in expense:
        cat_spend[t["category"]] += float(t["amount"])

    breakdown = {}
    for cat, amt in sorted(cat_spend.items(), key=lambda x: x[1], reverse=True):
        pct = (amt / total_spend * 100) if total_spend else 0.0
        breakdown[cat] = {"amount": round(amt, 2), "pct_of_spend": round(pct, 1)}
    return breakdown


def compute_income_concentration(transactions):
    """How dependent is the business on its biggest client(s)?
    top_client_pct = (largest single income source / total_income) * 100.
    A high value = revenue risk if that client leaves."""
    income, _ = split_income_expense(transactions)
    total_income = sum(abs(float(t["amount"])) for t in income)

    src_income = defaultdict(float)
    for t in income:
        src_income[t["merchant"]] += abs(float(t["amount"]))

    ranked = sorted(src_income.items(), key=lambda x: x[1], reverse=True)
    top_client_pct = (ranked[0][1] / total_income * 100) if total_income and ranked else 0.0
    top3_pct = (sum(v for _, v in ranked[:3]) / total_income * 100) if total_income else 0.0

    return {
        "top_client": ranked[0][0] if ranked else None,
        "top_client_pct": round(top_client_pct, 1),
        "top3_clients_pct": round(top3_pct, 1),
        "num_income_sources": len(ranked),
    }


def compute_fixed_vs_discretionary(transactions):
    """Committed monthly obligations vs discretionary spend.
    fixed_pct = (fixed_spend / total_spend) * 100.
    Tells an owner how much of their outflow is locked in regardless of
    revenue. (Category-based heuristic — see FIXED_CATEGORIES note above.)"""
    _, expense = split_income_expense(transactions)
    total_spend = sum(float(t["amount"]) for t in expense)

    fixed = sum(float(t["amount"]) for t in expense if t["category"] in FIXED_CATEGORIES)
    discretionary = total_spend - fixed

    return {
        "fixed_spend": round(fixed, 2),
        "discretionary_spend": round(discretionary, 2),
        "fixed_pct": round((fixed / total_spend * 100) if total_spend else 0.0, 1),
    }


def compute_burn_rate(transactions):
    """Spending pace over the statement period.
    daily_avg = total_spend / days_in_period.
    monthly_projection = daily_avg * 30.
    NOTE: true 'runway' (months of cash left) is NOT computable here — it
    needs a starting cash balance, which a transaction list doesn't contain.
    We report the burn rate and are honest that runway needs more data."""
    _, expense = split_income_expense(transactions)
    total_spend = sum(float(t["amount"]) for t in expense)

    dates = [parse_date(t["date"]) for t in transactions]
    valid = [d for d in dates if d is not None]
    bad_rows = len(dates) - len(valid)

    if len(valid) >= 2:
        days = (max(valid) - min(valid)).days or 1  # avoid div-by-zero
    else:
        days = 1

    daily_avg = total_spend / days
    return {
        "period_days": days,
        "daily_avg_spend": round(daily_avg, 2),
        "monthly_projection": round(daily_avg * 30, 2),
        "unparseable_dates": bad_rows,  # data-quality flag
        "runway_note": "Runway not computed: requires a starting cash balance.",
    }


def validate(kpis, transactions):
    """Cheap sanity checks so a broken computation fails loudly."""
    if kpis["total_spend"] <= 0:
        raise ValueError(f"total_spend <= 0 (got {kpis['total_spend']})")
    # category percentages should sum to ~100 (rounding tolerance)
    pct_sum = sum(v["pct_of_spend"] for v in kpis["spend_by_category"].values())
    if abs(pct_sum - 100) > 1.0:
        raise ValueError(f"category pct sum off: {pct_sum}")


def record_join_report(outputs_dir, report):
    """Add the join report to data_quality.json, keeping ingest's section."""
    dq_path = os.path.join(outputs_dir, "data_quality.json")
    dq = {}
    if os.path.exists(dq_path):
        with open(dq_path) as f:
            dq = json.load(f)
    dq["kpi_join"] = report
    with open(dq_path, "w") as f:
        json.dump(dq, f, indent=2)


def main(outputs_dir="outputs"):
    ingested = load_ingested(os.path.join(outputs_dir, "ingested.json"))
    categorized = load_transactions(os.path.join(outputs_dir, "categorized.json"))
    transactions, join_report = join_categories(ingested, categorized)
    transactions = apply_refunds(transactions)

    kpis = {}
    kpis.update(compute_core_kpis(transactions))          # the original 4
    kpis.update(compute_net_cash_flow(transactions))
    kpis["spend_by_category"] = compute_spend_by_category(transactions)
    kpis["income_concentration"] = compute_income_concentration(transactions)
    kpis["fixed_vs_discretionary"] = compute_fixed_vs_discretionary(transactions)
    kpis["burn_rate"] = compute_burn_rate(transactions)

    validate(kpis, transactions)

    kpis_path = os.path.join(outputs_dir, "kpis.json")
    with open(kpis_path, "w") as f:
        json.dump({"kpis": kpis}, f, indent=2)
    record_join_report(outputs_dir, join_report)
    left_out = join_report["excluded_total"]["rows"]
    print(f"KPIs saved to {kpis_path} ({join_report['rows_in_kpis']} rows used, "
          f"{left_out} left out at join -> data_quality.json)")
    return kpis


if __name__ == "__main__":
    main()