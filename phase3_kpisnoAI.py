"""
phase3_kpisnoAI.py — Deterministic KPI engine (NO LLM).

DESIGN DECISION (the headline of this project):
Every metric in this file is exact arithmetic on the transaction data, so it
is computed in plain Python and NEVER sent to a language model. The LLM in
this pipeline handles fuzzy work (categorization, summaries); money math has
to be correct and reproducible, so it lives here. That separation is
deliberate — an LLM's arithmetic can't be trusted for financial figures.

INPUT : outputs/categorized.json  (produced by the LLM categorizer)
        Each transaction: {date, merchant, amount, category}
        CONVENTION: negative amount = income, positive amount = expense.
OUTPUT: outputs/kpis.json
"""

import json
from collections import defaultdict
from datetime import datetime

# Categories treated as committed/fixed monthly obligations vs discretionary.
# NOTE: this is a category-based heuristic, not true recurrence detection.
# Detecting real month-over-month recurrence needs multi-month data, which a
# single statement doesn't give. Utilities here holds SaaS, rent, payroll,
# insurance, and bills — costs owed regardless of revenue.
FIXED_CATEGORIES = {"Utilities"}
DISCRETIONARY_CATEGORIES = {"Dining", "Shopping", "Other"}


def parse_date(raw):
    """Parse a statement date. Primary format MM-DD-YYYY; falls back to ISO.
    Returns a datetime, or None if unparseable (so bad rows are flagged, not
    silently dropped) — real statements contain malformed dates."""
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


def split_income_expense(transactions):
    """Convention: amount < 0 is income, amount > 0 is an expense."""
    income, expense = [], []
    for t in transactions:
        (income if float(t["amount"]) < 0 else expense).append(t)
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


def main():
    transactions = load_transactions()

    kpis = {}
    kpis.update(compute_core_kpis(transactions))          # the original 4
    kpis.update(compute_net_cash_flow(transactions))
    kpis["spend_by_category"] = compute_spend_by_category(transactions)
    kpis["income_concentration"] = compute_income_concentration(transactions)
    kpis["fixed_vs_discretionary"] = compute_fixed_vs_discretionary(transactions)
    kpis["burn_rate"] = compute_burn_rate(transactions)

    validate(kpis, transactions)

    with open("outputs/kpis.json", "w") as f:
        json.dump({"kpis": kpis}, f, indent=2)
    print("KPIs saved to outputs/kpis.json")
    return kpis


if __name__ == "__main__":
    main()