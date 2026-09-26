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
        Only {id, category} is taken from here, joined on id; the model
        never sees or returns amounts, dates or signs.
OUTPUT: outputs/kpis.json
        outputs/transactions.json  (every ingested row with its final
                                    category, status and reason)
        outputs/data_quality.json  ("kpi_join" section added)
"""

import json
import os
from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal

from afw.models import Category, Direction, Status, direction_allows

# Categories treated as committed/fixed monthly obligations vs discretionary.
# NOTE: this is a category-based heuristic, not true recurrence detection.
# Detecting real month-over-month recurrence needs multi-month data, which a
# single statement doesn't give. Utilities here holds SaaS, rent, payroll,
# insurance, and bills — costs owed regardless of revenue.
# (.value: rows carry plain strings, and a str-Enum member hashes by name.)
FIXED_CATEGORIES = {Category.UTILITIES.value}
DISCRETIONARY_CATEGORIES = {Category.DINING.value, Category.SHOPPING.value, Category.OTHER.value}

# Reasons a source row that passed ingest is still left out of the KPIs. The
# llm_* reasons and category_direction_mismatch are decided by the
# categorizer's reply validation (categorized.json "review"); the join also
# re-checks direction itself, so a stale or hand-edited file can't bypass it.
LLM_REVIEW_REASONS = ("llm_unparseable", "llm_missing", "llm_conflict", "llm_invalid_category",
                      "category_direction_mismatch")
JOIN_PROBLEMS = ("join_ambiguous", "join_missing", "category_invalid", "refund_original_excluded",
                 *LLM_REVIEW_REASONS)
CATEGORY_VALUES = {c.value for c in Category}

# KPI money math is exact Decimal; values are rounded half-up only when
# written out, and kpis.json keeps plain JSON numbers.
ZERO = Decimal(0)
CENT = Decimal("0.01")
TENTH = Decimal("0.1")


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


def join_categories(ingested, categorized, review=()):
    """Attach the LLM's category to each OK source row, by id. Returns
    (rows, report, results): the rows used for the KPIs, the join counts,
    and every ingested row with its final category, status and reason —
    the single per-row record written to transactions.json.

    Only ids of OK rows were ever sent to the model. An entry with any other
    id (or none) is ignored and counted. An OK row is left out of the KPIs
    (NEEDS_REVIEW, counted, never silently dropped) when the model returned
    no category for its id, conflicting categories for it, or a category
    that isn't in the Category enum. A row the categorizer put in `review`
    keeps that reason, and a category that contradicts the row's direction is
    category_direction_mismatch. A refund takes no category of its own: it
    is netted against its original DEBIT, and left out only if that DEBIT is.
    """
    report = Counter()
    review_reasons = {r["id"]: r["reason"] for r in review}
    sent_ids = {t["id"] for t in ingested if t["status"] == Status.OK.value}
    categories = defaultdict(set)
    for c in categorized:
        if c.get("id") in sent_ids:
            categories[c["id"]].add(str(c.get("category")))
        else:
            report["categorized_unknown_id"] += 1

    excluded = []

    def exclude(row, reason):
        excluded.append({**row, "status": Status.NEEDS_REVIEW.value, "reason": reason})
        report[reason] += 1

    joined = []
    for t in ingested:
        if t["status"] != Status.OK.value:
            continue  # ingest already marked it REJECTED / NEEDS_REVIEW and counted it
        row = {**t, "date": date.fromisoformat(t["date"])}
        if row["is_refund"]:
            joined.append(row)
            continue
        found = categories.get(row["id"], set())
        if row["id"] in review_reasons:
            exclude(row, review_reasons[row["id"]])
        elif not found:
            exclude(row, "join_missing")
        elif len(found) > 1:
            exclude(row, "join_ambiguous")
        elif (category := next(iter(found))) not in CATEGORY_VALUES:
            exclude(row, "category_invalid")
        elif not direction_allows(row["direction"], category):
            exclude(row, "category_direction_mismatch")
        else:
            joined.append({**row, "category": category})

    kept_debits = {r["id"]: r["category"] for r in joined if not r["is_refund"]}
    rows = []
    for row in joined:
        if not row["is_refund"]:
            rows.append(row)
        elif row["refund_of"] in kept_debits:
            rows.append({**row, "category": kept_debits[row["refund_of"]]})
        else:
            exclude(row, "refund_original_excluded")

    final = {r["id"]: (Status.OK.value, None, r["category"]) for r in rows}
    final.update({r["id"]: (r["status"], r["reason"], None) for r in excluded})
    results = []
    for t in ingested:
        status, reason, category = final.get(t["id"], (t["status"], t["reason"], None))
        results.append({**t, "category": category, "status": status, "reason": reason})

    excluded_amount = lambda d: str(sum((Decimal(r["amount"]) for r in excluded if r["direction"] == d),
                                        Decimal("0.00")))
    return rows, {
        "rows_in_kpis": len(rows),
        **{k: report[k] for k in JOIN_PROBLEMS},
        "categorized_unknown_id": report["categorized_unknown_id"],
        "excluded_total": {
            "debit": excluded_amount(Direction.DEBIT.value),
            "credit": excluded_amount(Direction.CREDIT.value),
            "rows": len(excluded),
        },
    }, results


def to_decimal(amount):
    """Exact value of an amount: ingested.json strings as-is, and numbers via
    their shortest repr (420.5 -> Decimal("420.5"), not the binary float)."""
    return amount if isinstance(amount, Decimal) else Decimal(str(amount))


def total(transactions):
    return sum((to_decimal(t["amount"]) for t in transactions), ZERO)


def money(value):
    """Exact Decimal -> the float written to kpis.json, rounded half-up to cents."""
    return float(value.quantize(CENT, ROUND_HALF_UP))


def pct(part, whole):
    """part / whole as a percentage, rounded half-up to 0.1; 0.0 when whole is 0."""
    return float((part / whole * 100).quantize(TENTH, ROUND_HALF_UP)) if whole else 0.0


def ranked(amounts):
    """(name, amount) pairs, largest first; ties broken by name so the result
    doesn't depend on row order."""
    return sorted(amounts.items(), key=lambda x: (-x[1], x[0]))


def apply_refunds(transactions):
    """Net each refund against the DEBIT it refunds, so a returned purchase
    lowers that DEBIT's spend (same category, same merchant) instead of
    counting as income. Refund rows stay in the list; split_income_expense
    skips them."""
    refunded = defaultdict(lambda: ZERO)
    for t in transactions:
        if t.get("is_refund"):
            refunded[t["refund_of"]] += to_decimal(t["amount"])
    return [{**t, "amount": to_decimal(t["amount"]) - refunded[t["id"]]} if t.get("id") in refunded else t
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

    total_spend = total(expense)
    total_income = total(income)

    # top_merchants = 3 merchants with the highest total spend
    merchant_spend = defaultdict(lambda: ZERO)
    for t in expense:
        merchant_spend[t["merchant"]] += to_decimal(t["amount"])
    top_merchants = [m for m, _ in ranked(merchant_spend)[:3]]

    # average_expense = total_spend / number of expense transactions
    average_expense = total_spend / len(expense) if expense else ZERO

    return {
        "total_spend": money(total_spend),
        "total_income": money(total_income),
        "top_merchants": top_merchants,
        "average_expense": money(average_expense),
    }


def compute_net_cash_flow(transactions):
    """net_cash_flow = total_income - total_spend.
    The single most important 'am I okay' number: did more money come in than
    went out this period. Positive = surplus, negative = burning cash.
    Status is decided on the exact result, before rounding."""
    income, expense = split_income_expense(transactions)
    net = total(income) - total(expense)
    return {
        "net_cash_flow": money(net),
        "status": "surplus" if net >= 0 else "deficit",
    }


def compute_spend_by_category(transactions):
    """spend per category, and each as a % of total spend.
    pct = (category_spend / total_spend) * 100.
    This is where an owner finds costs to cut."""
    _, expense = split_income_expense(transactions)
    total_spend = total(expense)

    cat_spend = defaultdict(lambda: ZERO)
    for t in expense:
        cat_spend[t["category"]] += to_decimal(t["amount"])

    return {cat: {"amount": money(amt), "pct_of_spend": pct(amt, total_spend)}
            for cat, amt in ranked(cat_spend)}


def compute_income_concentration(transactions):
    """How dependent is the business on its biggest client(s)?
    top_client_pct = (largest single income source / total_income) * 100.
    A high value = revenue risk if that client leaves."""
    income, _ = split_income_expense(transactions)
    total_income = total(income)

    src_income = defaultdict(lambda: ZERO)
    for t in income:
        src_income[t["merchant"]] += to_decimal(t["amount"])

    sources = ranked(src_income)
    return {
        "top_client": sources[0][0] if sources else None,
        "top_client_pct": pct(sources[0][1], total_income) if sources else 0.0,
        "top3_clients_pct": pct(sum((v for _, v in sources[:3]), ZERO), total_income),
        "num_income_sources": len(sources),
    }


def compute_fixed_vs_discretionary(transactions):
    """Committed monthly obligations vs discretionary spend.
    fixed_pct = (fixed_spend / total_spend) * 100.
    Tells an owner how much of their outflow is locked in regardless of
    revenue. (Category-based heuristic — see FIXED_CATEGORIES note above.)"""
    _, expense = split_income_expense(transactions)
    total_spend = total(expense)
    fixed = total(t for t in expense if t["category"] in FIXED_CATEGORIES)

    return {
        "fixed_spend": money(fixed),
        "discretionary_spend": money(total_spend - fixed),
        "fixed_pct": pct(fixed, total_spend),
    }


def compute_burn_rate(transactions):
    """Spending pace over the statement period.
    daily_avg = total_spend / days_in_period.
    monthly_projection = daily_avg * 30.
    NOTE: true 'runway' (months of cash left) is NOT computable here — it
    needs a starting cash balance, which a transaction list doesn't contain.
    We report the burn rate and are honest that runway needs more data."""
    _, expense = split_income_expense(transactions)
    total_spend = total(expense)

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
        "daily_avg_spend": money(daily_avg),
        "monthly_projection": money(daily_avg * 30),
        "unparseable_dates": bad_rows,  # data-quality flag
        "runway_note": "Runway not computed: requires a starting cash balance.",
    }


def validate(kpis, transactions):
    """Cheap sanity checks so a broken computation fails loudly. Zero spend
    is valid (every row refunded or left out at the join)."""
    if kpis["total_spend"] < 0:
        raise ValueError(f"total_spend < 0 (got {kpis['total_spend']})")
    # category percentages should sum to ~100 (rounding tolerance)
    pct_sum = sum(v["pct_of_spend"] for v in kpis["spend_by_category"].values())
    if kpis["total_spend"] > 0 and abs(pct_sum - 100) > 1.0:
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
    with open(os.path.join(outputs_dir, "categorized.json")) as f:
        categorizer_output = json.load(f)
    transactions, join_report, results = join_categories(
        ingested, categorizer_output["categorized"], categorizer_output.get("review", []))
    join_report["llm"] = categorizer_output.get("llm", {})
    # Written first, so the record of what was left out survives a KPI failure.
    record_join_report(outputs_dir, join_report)
    with open(os.path.join(outputs_dir, "transactions.json"), "w") as f:
        json.dump({"transactions": results}, f, indent=2)
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
    left_out = join_report["excluded_total"]["rows"]
    print(f"KPIs saved to {kpis_path} ({join_report['rows_in_kpis']} rows used, "
          f"{left_out} left out at join -> data_quality.json)")
    return kpis


if __name__ == "__main__":
    main()