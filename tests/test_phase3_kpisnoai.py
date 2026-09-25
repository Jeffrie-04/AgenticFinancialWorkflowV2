"""
Tests for phase3_kpisnoAI.py — the deterministic KPI engine.

tests/fixtures/categorized_sample.json is a frozen snapshot of a real
outputs/categorized.json produced from data/transactiondata.csv (50
transactions, including one row with an unparseable date — "2024-03" for
the AWS transaction — which is a genuine data-quality issue coming out of
the LLM categorizer, not a fixture typo). Using a frozen copy means these
tests don't depend on outputs/categorized.json changing between pipeline
runs, and don't touch the user's real outputs/ files.

The expected numbers for the frozen fixture were independently computed
with pandas groupby/sum (a different code path than the module under
test) so this isn't just re-deriving the same formulas twice — see the
values in FullFixtureExpected. The KPI functions now split on `direction`
rather than sign; directed() converts the legacy signed rows, and every
expected number is unchanged.

The join/refund tests at the bottom build ingested.json rows (the output
of afw/ingest.py) and categorized.json rows (LLM output) directly.
"""
import itertools
import json
import os
from datetime import date
from typing import ClassVar

import pytest

import phase3_kpisnoAI as kpis_mod

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "categorized_sample.json")


def directed(transactions):
    """Legacy signed rows -> KPI input rows: negative = CREDIT, positive =
    DEBIT, amount made positive (what ingest produces)."""
    out = []
    for t in transactions:
        amount = float(t["amount"])
        out.append({**t, "direction": "CREDIT" if amount < 0 else "DEBIT", "amount": abs(amount)})
    return out


@pytest.fixture
def full_fixture():
    with open(FIXTURE_PATH) as f:
        return directed(json.load(f)["categorized"])


# Independently computed (via pandas) ground truth for the frozen fixture.
class FullFixtureExpected:
    total_spend = 8979.42
    total_income = 14700.0
    average_expense = 204.08
    top_merchants: ClassVar[list[str]] = ["Square Payroll", "Best Buy", "WeWork"]
    net_cash_flow = 5720.58
    status = "surplus"
    spend_by_category: ClassVar[dict[str, dict[str, float]]] = {
        "Utilities": {"amount": 5418.76, "pct_of_spend": 60.3},
        "Shopping": {"amount": 2638.77, "pct_of_spend": 29.4},
        "Other": {"amount": 502.24, "pct_of_spend": 5.6},
        "Dining": {"amount": 419.65, "pct_of_spend": 4.7},
    }
    top_client = "Retainer Payment - Law Firm"
    top_client_pct = 30.6
    top3_clients_pct = 73.5
    num_income_sources = 6
    fixed_spend = 5418.76
    discretionary_spend = 3560.66
    fixed_pct = 60.3
    period_days = 23
    daily_avg_spend = 390.41
    monthly_projection = 11712.29
    unparseable_dates = 1


# ---------------------------------------------------------------- parse_date

class TestParseDate:
    def test_primary_format(self):
        d = kpis_mod.parse_date("10-01-2024")
        assert (d.year, d.month, d.day) == (2024, 10, 1)

    def test_iso_fallback(self):
        d = kpis_mod.parse_date("2024-10-01")
        assert (d.year, d.month, d.day) == (2024, 10, 1)

    def test_unparseable_returns_none(self):
        assert kpis_mod.parse_date("not-a-date") is None

    def test_real_bad_row_from_fixture_returns_none(self):
        # The actual malformed date that shows up in the frozen fixture.
        assert kpis_mod.parse_date("2024-03") is None

    def test_date_objects_pass_through(self):
        assert kpis_mod.parse_date(date(2024, 10, 1)) == date(2024, 10, 1)


# ------------------------------------------------------------ load_transactions

class TestLoadTransactions:
    def test_loads_fixture(self):
        transactions = kpis_mod.load_transactions(FIXTURE_PATH)
        assert len(transactions) == 50
        assert set(transactions[0].keys()) >= {"date", "merchant", "amount", "category"}


# -------------------------------------------------------- split_income_expense

class TestSplitIncomeExpense:
    def test_splits_purely_on_direction(self):
        transactions = [
            {"amount": 100, "direction": "CREDIT", "category": "Shopping"},  # direction wins over category
            {"amount": 50, "direction": "DEBIT", "category": "Income"},
            {"amount": 1, "direction": "CREDIT", "category": "Other"},
        ]
        income, expense = kpis_mod.split_income_expense(transactions)
        assert income == [transactions[0], transactions[2]]
        assert expense == [transactions[1]]

    def test_refunds_are_neither_income_nor_expense(self):
        transactions = [
            {"amount": 50, "direction": "CREDIT", "is_refund": True},
            {"amount": 10, "direction": "CREDIT", "is_refund": False},
        ]
        income, expense = kpis_mod.split_income_expense(transactions)
        assert income == [transactions[1]]
        assert expense == []

    def test_string_amounts_are_coerced(self):
        transactions = [{"merchant": "A", "amount": "5.00", "direction": "CREDIT"},
                        {"merchant": "B", "amount": "5.00", "direction": "DEBIT"}]
        result = kpis_mod.compute_core_kpis(transactions)
        assert (result["total_income"], result["total_spend"]) == (5.0, 5.0)


# ----------------------------------------------------------- compute_core_kpis

class TestComputeCoreKpis:
    def test_full_fixture_matches_independent_calculation(self, full_fixture):
        result = kpis_mod.compute_core_kpis(full_fixture)
        assert result["total_spend"] == FullFixtureExpected.total_spend
        assert result["total_income"] == FullFixtureExpected.total_income
        assert result["average_expense"] == FullFixtureExpected.average_expense
        assert result["top_merchants"] == FullFixtureExpected.top_merchants

    def test_top_merchants_ranks_by_summed_spend_not_transaction_count(self):
        transactions = directed([
            {"merchant": "A", "amount": 10},
            {"merchant": "A", "amount": 10},
            {"merchant": "A", "amount": 10},  # A totals 30 across 3 small txns
            {"merchant": "B", "amount": 40},  # B totals 40 in a single txn
        ])
        result = kpis_mod.compute_core_kpis(transactions)
        assert result["top_merchants"][0] == "B"

    def test_no_expenses_gives_zero_average_and_empty_merchants(self):
        transactions = directed([{"merchant": "X", "amount": -100}])
        result = kpis_mod.compute_core_kpis(transactions)
        assert result["total_spend"] == 0
        assert result["average_expense"] == 0.0
        assert result["top_merchants"] == []


# ------------------------------------------------------- compute_net_cash_flow

class TestComputeNetCashFlow:
    def test_full_fixture_is_a_surplus(self, full_fixture):
        result = kpis_mod.compute_net_cash_flow(full_fixture)
        assert result["net_cash_flow"] == FullFixtureExpected.net_cash_flow
        assert result["status"] == FullFixtureExpected.status

    def test_deficit_when_spend_exceeds_income(self):
        transactions = directed([{"amount": -10}, {"amount": 100}])
        result = kpis_mod.compute_net_cash_flow(transactions)
        assert result["net_cash_flow"] == -90.0
        assert result["status"] == "deficit"

    def test_zero_net_counts_as_surplus(self):
        transactions = directed([{"amount": -50}, {"amount": 50}])
        result = kpis_mod.compute_net_cash_flow(transactions)
        assert result["net_cash_flow"] == 0.0
        assert result["status"] == "surplus"


# --------------------------------------------------- compute_spend_by_category

class TestComputeSpendByCategory:
    def test_full_fixture_matches_independent_calculation(self, full_fixture):
        result = kpis_mod.compute_spend_by_category(full_fixture)
        assert result == FullFixtureExpected.spend_by_category

    def test_percentages_sum_to_roughly_100(self, full_fixture):
        result = kpis_mod.compute_spend_by_category(full_fixture)
        pct_sum = sum(v["pct_of_spend"] for v in result.values())
        assert abs(pct_sum - 100) <= 1.0

    def test_no_expenses_returns_empty_dict_without_dividing_by_zero(self):
        transactions = directed([{"amount": -100, "category": "Income"}])
        result = kpis_mod.compute_spend_by_category(transactions)
        assert result == {}


# ----------------------------------------------- compute_income_concentration

class TestComputeIncomeConcentration:
    def test_full_fixture_matches_independent_calculation(self, full_fixture):
        result = kpis_mod.compute_income_concentration(full_fixture)
        assert result["top_client"] == FullFixtureExpected.top_client
        assert result["top_client_pct"] == FullFixtureExpected.top_client_pct
        assert result["top3_clients_pct"] == FullFixtureExpected.top3_clients_pct
        assert result["num_income_sources"] == FullFixtureExpected.num_income_sources

    def test_no_income_returns_safe_defaults(self):
        transactions = directed([{"merchant": "A", "amount": 100}])
        result = kpis_mod.compute_income_concentration(transactions)
        assert result == {
            "top_client": None,
            "top_client_pct": 0.0,
            "top3_clients_pct": 0.0,
            "num_income_sources": 0,
        }


# ------------------------------------------- compute_fixed_vs_discretionary

class TestComputeFixedVsDiscretionary:
    def test_full_fixture_matches_independent_calculation(self, full_fixture):
        result = kpis_mod.compute_fixed_vs_discretionary(full_fixture)
        assert result["fixed_spend"] == FullFixtureExpected.fixed_spend
        assert result["discretionary_spend"] == FullFixtureExpected.discretionary_spend
        assert result["fixed_pct"] == FullFixtureExpected.fixed_pct

    def test_fixed_plus_discretionary_equals_total_spend(self, full_fixture):
        result = kpis_mod.compute_fixed_vs_discretionary(full_fixture)
        assert round(result["fixed_spend"] + result["discretionary_spend"], 2) == FullFixtureExpected.total_spend

    def test_all_discretionary_categories_gives_zero_fixed(self):
        transactions = directed([
            {"amount": 50, "category": "Dining"},
            {"amount": 25, "category": "Shopping"},
        ])
        result = kpis_mod.compute_fixed_vs_discretionary(transactions)
        assert result["fixed_spend"] == 0
        assert result["fixed_pct"] == 0.0


# ------------------------------------------------------- compute_burn_rate

class TestComputeBurnRate:
    def test_full_fixture_matches_independent_calculation(self, full_fixture):
        result = kpis_mod.compute_burn_rate(full_fixture)
        assert result["period_days"] == FullFixtureExpected.period_days
        assert result["daily_avg_spend"] == FullFixtureExpected.daily_avg_spend
        assert result["monthly_projection"] == FullFixtureExpected.monthly_projection
        assert result["unparseable_dates"] == FullFixtureExpected.unparseable_dates

    def test_single_day_of_transactions_falls_back_to_one_day(self):
        transactions = directed([
            {"amount": 100, "date": "10-01-2024"},
            {"amount": 50, "date": "10-01-2024"},
        ])
        result = kpis_mod.compute_burn_rate(transactions)
        assert result["period_days"] == 1
        assert result["daily_avg_spend"] == 150.0

    def test_fewer_than_two_valid_dates_falls_back_to_one_day(self):
        transactions = directed([
            {"amount": 100, "date": "not-a-date"},
            {"amount": 50, "date": "also-bad"},
        ])
        result = kpis_mod.compute_burn_rate(transactions)
        assert result["period_days"] == 1
        assert result["unparseable_dates"] == 2


# --------------------------------------------------------------- validate

class TestValidate:
    def test_full_fixture_passes(self, full_fixture):
        full_kpis = {}
        full_kpis.update(kpis_mod.compute_core_kpis(full_fixture))
        full_kpis.update(kpis_mod.compute_net_cash_flow(full_fixture))
        full_kpis["spend_by_category"] = kpis_mod.compute_spend_by_category(full_fixture)
        kpis_mod.validate(full_kpis, full_fixture)  # should not raise

    def test_raises_on_negative_total_spend(self):
        with pytest.raises(ValueError, match="total_spend"):
            kpis_mod.validate({"total_spend": -1, "spend_by_category": {}}, [])

    def test_zero_spend_is_valid(self):
        kpis_mod.validate({"total_spend": 0, "spend_by_category": {}}, [])
        kpis_mod.validate({"total_spend": 0,
                           "spend_by_category": {"Shopping": {"amount": 0.0, "pct_of_spend": 0.0}}}, [])

    def test_raises_when_category_percentages_dont_sum_to_100(self):
        bad_kpis = {
            "total_spend": 100,
            "spend_by_category": {
                "Shopping": {"amount": 100, "pct_of_spend": 50.0},  # should be ~100
            },
        }
        with pytest.raises(ValueError, match="category pct sum off"):
            kpis_mod.validate(bad_kpis, [])


# ============================================ exact math, deterministic ranks

def row(merchant, amount, direction, **extra):
    return {"merchant": merchant, "amount": amount, "direction": direction, **extra}


def numeric_leaves(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from numeric_leaves(v)
    elif isinstance(value, list):
        for v in value:
            yield from numeric_leaves(v)
    elif not isinstance(value, (str, bool)) and value is not None:
        yield value


class TestExactMath:
    @pytest.mark.parametrize("transactions", [
        # 0.10 + 0.20 in, 0.30 out
        [row("A", "0.10", "CREDIT"), row("B", "0.20", "CREDIT"), row("C", "0.30", "DEBIT")],
        # 0.30 in, 0.10 + 0.20 out: float gives net -5.5e-17 -> "deficit"
        [row("A", "0.30", "CREDIT"), row("B", "0.10", "DEBIT"), row("C", "0.20", "DEBIT")],
        # Same with float amounts (legacy rows): must convert via repr, since
        # Decimal(0.3) - Decimal(0.1) - Decimal(0.2) is -2.8e-17.
        [row("A", 0.1, "CREDIT"), row("B", 0.2, "CREDIT"), row("C", 0.3, "DEBIT")],
        [row("A", 0.3, "CREDIT"), row("B", 0.1, "DEBIT"), row("C", 0.2, "DEBIT")],
    ])
    def test_status_decided_from_exact_net(self, transactions):
        result = kpis_mod.compute_net_cash_flow(transactions)
        assert result == {"net_cash_flow": 0.0, "status": "surplus"}

    @pytest.mark.parametrize("refunds", [("0.10", "0.20", "0.30"), ("0.30", "0.20", "0.10")])
    def test_refunds_accumulate_exactly_in_any_order(self, refunds):
        transactions = [row("Shop", "0.60", "DEBIT", id="d", category="Shopping")] + [
            row("Shop", amt, "CREDIT", id=f"r{i}", is_refund=True, refund_of="d")
            for i, amt in enumerate(refunds)]
        netted = kpis_mod.apply_refunds(transactions)
        assert netted[0]["amount"] == 0  # exactly, not 1e-16 off
        assert kpis_mod.compute_core_kpis(netted)["total_spend"] == 0.0

    def test_rounding_is_half_up(self):
        # average of $0.02 and $0.03 is exactly $0.025: half-up -> 0.03, half-even -> 0.02
        transactions = [row("A", "0.02", "DEBIT"), row("B", "0.03", "DEBIT")]
        assert kpis_mod.compute_core_kpis(transactions)["average_expense"] == 0.03

    def test_kpi_output_is_plain_numbers(self, tmp_path):
        kpis, _ = run_main(tmp_path, [RENT[0], ing("c", "Client", "0.10", "CREDIT")],
                           [RENT[1], cat("Client", -0.1, "Income")])
        leaves = list(numeric_leaves(kpis))
        assert leaves and all(type(v) in (int, float) for v in leaves)


FOUR_TIED = [row(m, "1.00", "DEBIT", category="Other") for m in ("Delta", "Charlie", "Bravo", "Alpha")]


class TestDeterministicRanking:
    @pytest.mark.parametrize("order", [FOUR_TIED, FOUR_TIED[::-1]])
    def test_top_merchant_ties_broken_by_name(self, order):
        assert kpis_mod.compute_core_kpis(order)["top_merchants"] == ["Alpha", "Bravo", "Charlie"]

    @pytest.mark.parametrize("order", [["Zed", "Amy"], ["Amy", "Zed"]])
    def test_top_client_ties_broken_by_name(self, order):
        transactions = [row(m, "500.00", "CREDIT") for m in order]
        assert kpis_mod.compute_income_concentration(transactions)["top_client"] == "Amy"


# ================================================ join with ingested.json

def ing(row_id, merchant, amount, direction, day="2024-10-03", status="OK", reason=None,
        is_refund=False, refund_of=None):
    """One row as afw/ingest.py writes it to ingested.json."""
    usable = status != "REJECTED"
    return {
        "id": row_id, "source_file": "t.csv", "source_row": 2, "account_id": "default",
        "date": day if usable else None, "amount": amount if usable else None,
        "direction": direction if usable else None, "merchant": merchant, "description": "",
        "currency": "USD", "is_transfer": False, "is_refund": is_refund, "refund_of": refund_of,
        "status": status, "reason": reason,
    }


def cat(merchant, amount, category, day="2024-10-03"):
    """One row as the LLM categorizer writes it to categorized.json."""
    return {"date": day, "merchant": merchant, "amount": amount, "category": category}


def run_main(tmp_path, ingested, categorized):
    (tmp_path / "ingested.json").write_text(json.dumps({"source_file": "t.csv", "transactions": ingested}))
    (tmp_path / "categorized.json").write_text(json.dumps({"categorized": categorized}))
    kpis = kpis_mod.main(outputs_dir=str(tmp_path))
    dq = json.loads((tmp_path / "data_quality.json").read_text())
    return kpis, dq["kpi_join"]


RENT = (ing("rent", "Landlord", "1000.00", "DEBIT"), cat("Landlord", 1000.0, "Utilities"))

# $100 purchase, refunded $20 then $30 on later days -> $50 net spend.
REFUND_20_30 = [
    ing("d", "Shop", "100.00", "DEBIT", day="2024-10-01"),
    ing("r1", "Shop", "20.00", "CREDIT", day="2024-10-02", is_refund=True, refund_of="d"),
    ing("r2", "Shop", "30.00", "CREDIT", day="2024-10-03", is_refund=True, refund_of="d"),
]
REFUND_20_30_LLM = [
    cat("Shop", 100.0, "Shopping", day="2024-10-01"),
    cat("Shop", -20.0, "Income", day="2024-10-02"),
    cat("Shop", -30.0, "Income", day="2024-10-03"),
]


class TestJoinCategories:
    @pytest.mark.parametrize("llm_date", ["2024-10-03", "10-03-2024"])
    def test_f1_category_joined_across_date_formats(self, llm_date):
        rows, report = kpis_mod.join_categories(
            [ing("a", "Shop", "10.00", "DEBIT")], [cat("Shop", 10.0, "Shopping", day=llm_date)])
        assert [(r["id"], r["category"]) for r in rows] == [("a", "Shopping")]
        assert report["rows_in_kpis"] == 1

    def test_join_uses_normalized_merchant_and_ignores_llm_sign(self):
        rows, _ = kpis_mod.join_categories(
            [ing("a", "Home Depot", "10.00", "DEBIT")], [cat("  HOME   DEPOT", -10.0, "Shopping")])
        assert rows[0]["category"] == "Shopping"
        assert rows[0]["merchant"] == "Home Depot"  # merchant comes from source data

    def test_f2_identical_source_rows_are_ambiguous(self):
        rows, report = kpis_mod.join_categories(
            [ing("a", "Starbucks", "4.50", "DEBIT"), ing("b", "Starbucks", "4.50", "DEBIT")],
            [cat("Starbucks", 4.5, "Dining"), cat("Starbucks", 4.5, "Dining")])
        assert rows == []
        assert report["join_ambiguous"] == 2
        assert report["excluded_total"] == {"debit": "9.00", "credit": "0.00", "rows": 2}

    def test_f3_missing_category_is_needs_review(self):
        rows, report = kpis_mod.join_categories([ing("a", "Shop", "10.00", "DEBIT")], [])
        assert rows == []
        assert report["join_missing"] == 1

    def test_f4_llm_invented_row_is_ignored_and_counted(self):
        rows, report = kpis_mod.join_categories(
            [ing("a", "Shop", "10.00", "DEBIT")],
            [cat("Shop", 10.0, "Shopping"), cat("Ghost Vendor", 999.0, "Shopping")])
        assert [r["id"] for r in rows] == ["a"]
        assert report["categorized_unmatched"] == 1

    def test_f5_rejected_and_ingest_review_rows_excluded(self):
        rows, report = kpis_mod.join_categories(
            [ing("a", "Shop", "10.00", "DEBIT"),
             ing("r", "Verizon", None, None, status="REJECTED", reason="amount_zero"),
             ing("n", "", "55.00", "DEBIT", status="NEEDS_REVIEW", reason="merchant_blank")],
            [cat("Shop", 10.0, "Shopping"), cat("", 55.0, "Other")])
        assert [r["id"] for r in rows] == ["a"]
        # Ingest-level statuses are already counted in data_quality.json by ingest.
        assert report["join_missing"] == report["join_ambiguous"] == 0

    def test_invalid_llm_category_is_needs_review(self):
        rows, report = kpis_mod.join_categories(
            [ing("a", "Shop", "10.00", "DEBIT")], [cat("Shop", 10.0, "Groceries")])
        assert rows == []
        assert report["category_invalid"] == 1

    def test_duplicated_llm_rows_that_disagree_are_ambiguous(self):
        rows, report = kpis_mod.join_categories(
            [ing("a", "Shop", "10.00", "DEBIT")],
            [cat("Shop", 10.0, "Shopping"), cat("Shop", 10.0, "Other")])
        assert rows == []
        assert report["join_ambiguous"] == 1

    def test_unparseable_llm_row_is_counted(self):
        rows, report = kpis_mod.join_categories(
            [ing("a", "Shop", "10.00", "DEBIT")],
            [cat("Shop", 10.0, "Shopping"), cat("Shop", "ten", "Shopping")])
        assert [r["id"] for r in rows] == ["a"]
        assert report["categorized_invalid"] == 1

    def test_f8_refund_of_excluded_debit_is_needs_review(self):
        rows, report = kpis_mod.join_categories(
            [ing("d", "Home Depot", "200.00", "DEBIT"),  # no categorized match -> excluded
             ing("r", "Home Depot", "50.00", "CREDIT", day="2024-10-05", is_refund=True, refund_of="d")],
            [cat("Home Depot", -50.0, "Income", day="2024-10-05")])
        assert rows == []
        assert report["join_missing"] == 1
        assert report["refund_original_excluded"] == 1
        assert report["excluded_total"] == {"debit": "200.00", "credit": "50.00", "rows": 2}


class TestJoinGroups:
    """Rows sharing (date, normalized merchant) are judged together: if the
    LLM lost or mis-echoed one of them, its categories for the others can't
    be trusted either."""

    def test_codex_paper_meal_group_with_missing_row_is_inconsistent(self):
        # Same store, same day: $10 paper and $20 meal. The LLM echoed the
        # paper row with the meal's amount and dropped the other, so a
        # key-only join would give the $20 meal the paper's category.
        rows, report = kpis_mod.join_categories(
            [ing("paper", "Costco", "10.00", "DEBIT"), ing("meal", "Costco", "20.00", "DEBIT")],
            [cat("Costco", 20.0, "Shopping")])
        assert rows == []
        assert report["join_missing"] == 1             # the $10 row keeps its own reason
        assert report["join_group_inconsistent"] == 1  # the $20 row is pulled out with it
        assert report["excluded_total"] == {"debit": "30.00", "credit": "0.00", "rows": 2}

    def test_group_with_missing_row_is_inconsistent_even_when_counts_match(self):
        # Two source rows, two LLM rows, but the LLM echoed $10 as $25: the
        # counts reconcile, so only the missing-row rule catches it.
        rows, report = kpis_mod.join_categories(
            [ing("paper", "Costco", "10.00", "DEBIT"), ing("meal", "Costco", "20.00", "DEBIT")],
            [cat("Costco", 25.0, "Shopping"), cat("Costco", 20.0, "Dining")])
        assert rows == []
        assert report["join_missing"] == 1
        assert report["join_group_inconsistent"] == 1

    def test_group_with_extra_llm_row_is_inconsistent(self):
        rows, report = kpis_mod.join_categories(
            [ing("paper", "Costco", "10.00", "DEBIT"), ing("meal", "Costco", "20.00", "DEBIT")],
            [cat("Costco", 10.0, "Shopping"), cat("Costco", 20.0, "Dining"), cat("Costco", 30.0, "Other")])
        assert rows == []
        assert report["join_group_inconsistent"] == 2
        assert report["categorized_unmatched"] == 1

    def test_consistent_group_joins_normally(self):
        rows, report = kpis_mod.join_categories(
            [ing("paper", "Costco", "10.00", "DEBIT"), ing("meal", "Costco", "20.00", "DEBIT")],
            [cat("Costco", 20.0, "Dining"), cat("Costco", 10.0, "Shopping")])
        assert [(r["id"], r["category"]) for r in rows] == [("paper", "Shopping"), ("meal", "Dining")]
        assert report["join_group_inconsistent"] == 0

    def test_echoed_rejected_row_makes_its_group_inconsistent(self):
        # Pins accepted choice 4: the LLM saw the raw CSV, so it echoes a row
        # ingest REJECTED (here amount_zero). Its "0" parses, lands in the
        # (date, merchant) group, and the group's LLM count exceeds its
        # purchases -> the good row in that group is left out too.
        rows, report = kpis_mod.join_categories(
            [RENT[0], ing("z", "Landlord", None, None, status="REJECTED", reason="amount_zero")],
            [RENT[1], cat("Landlord", 0, "Utilities")])
        assert rows == []
        assert report["join_group_inconsistent"] == 1
        assert report["categorized_unmatched"] == 1
        assert report["excluded_total"] == {"debit": "1000.00", "credit": "0.00", "rows": 1}

    def test_group_problem_does_not_spread_to_other_days_or_merchants(self):
        rows, report = kpis_mod.join_categories(
            [ing("a", "Costco", "10.00", "DEBIT"),
             ing("b", "Costco", "20.00", "DEBIT", day="2024-10-04"),
             ing("c", "Staples", "30.00", "DEBIT")],
            [cat("Costco", 20.0, "Shopping", day="2024-10-04"), cat("Staples", 30.0, "Shopping")])
        assert [r["id"] for r in rows] == ["b", "c"]
        assert report["join_missing"] == 1
        assert report["join_group_inconsistent"] == 0


class TestRefundNetting:
    def test_same_day_full_refund_nets_to_zero_not_ambiguous(self, tmp_path):
        # DEBIT and refund share (date, merchant, |amount|); refunds are left
        # out of the source-side uniqueness check, and the refund's "Income"
        # label doesn't make the DEBIT's category ambiguous.
        kpis, report = run_main(tmp_path, [
            ing("d", "Shop", "100.00", "DEBIT"),
            ing("r", "Shop", "100.00", "CREDIT", is_refund=True, refund_of="d"),
        ], [cat("Shop", 100.0, "Shopping"), cat("Shop", -100.0, "Income")])
        assert report["join_ambiguous"] == 0
        assert kpis["total_spend"] == 0.0
        assert kpis["total_income"] == 0.0
        assert kpis["spend_by_category"] == {"Shopping": {"amount": 0.0, "pct_of_spend": 0.0}}

    def test_partial_refunds_net_to_remaining_spend(self, tmp_path):
        kpis, _ = run_main(tmp_path, REFUND_20_30, REFUND_20_30_LLM)
        assert kpis["total_spend"] == 50.0
        assert kpis["total_income"] == 0.0

    def test_full_refund_on_a_later_day_nets_to_zero(self, tmp_path):
        kpis, _ = run_main(tmp_path, [
            ing("d", "Shop", "100.00", "DEBIT", day="2024-10-01"),
            ing("r", "Shop", "100.00", "CREDIT", day="2024-10-09", is_refund=True, refund_of="d"),
        ], [cat("Shop", 100.0, "Shopping", day="2024-10-01")])
        assert (kpis["total_spend"], kpis["total_income"]) == (0.0, 0.0)

    def test_refund_order_does_not_change_kpis(self, tmp_path):
        results = []
        for order in itertools.permutations(REFUND_20_30):
            out = tmp_path / str(len(results))
            out.mkdir()
            results.append(run_main(out, list(order), REFUND_20_30_LLM)[0])
        assert all(r == results[0] for r in results)
        assert results[0]["total_spend"] == 50.0


    def test_f6_refund_nets_against_original_debit(self, tmp_path):
        kpis, _ = run_main(tmp_path, [
            RENT[0],
            ing("d", "Home Depot", "200.00", "DEBIT"),
            ing("r", "Home Depot", "50.00", "CREDIT", is_refund=True, refund_of="d"),
        ], [RENT[1], cat("Home Depot", 200.0, "Shopping"), cat("Home Depot", -50.0, "Income")])
        assert kpis["total_spend"] == 1150.0
        assert kpis["total_income"] == 0.0
        assert kpis["spend_by_category"]["Shopping"]["amount"] == 150.0  # the DEBIT's category
        assert "Income" not in kpis["spend_by_category"]
        assert kpis["average_expense"] == 575.0  # net spend / 2 DEBIT rows

    def test_refund_counts_without_its_own_llm_row(self, tmp_path):
        kpis, report = run_main(tmp_path, [
            RENT[0],
            ing("d", "Home Depot", "200.00", "DEBIT"),
            ing("r", "Home Depot", "50.00", "CREDIT", is_refund=True, refund_of="d"),
        ], [RENT[1], cat("Home Depot", 200.0, "Shopping")])
        assert kpis["total_spend"] == 1150.0
        assert report["join_missing"] == 0


class TestCodexAcceptance:
    def test_rejected_row_and_unsourced_llm_row_contribute_zero(self, tmp_path):
        baseline, _ = run_main(tmp_path, [RENT[0]], [RENT[1]])
        kpis, report = run_main(tmp_path, [
            RENT[0],
            ing("r", "Verizon", None, None, status="REJECTED", reason="amount_unparseable"),
        ], [RENT[1], cat("Verizon", 999.0, "Utilities"), cat("Ghost", -5000.0, "Income")])
        assert kpis == baseline
        assert report["categorized_unmatched"] == 2

    def test_100_debit_plus_50_refund(self, tmp_path):
        kpis, _ = run_main(tmp_path, [
            ing("d", "Shop", "100.00", "DEBIT", day="2024-10-01"),
            ing("r", "Shop", "50.00", "CREDIT", day="2024-10-02", is_refund=True, refund_of="d"),
        ], [cat("Shop", 100.0, "Shopping", day="2024-10-01"), cat("Shop", -50.0, "Income", day="2024-10-02")])
        assert kpis["total_spend"] == 50.0
        assert kpis["total_income"] == 0.0

    def test_credit_duplicated_in_categorized_counts_once(self, tmp_path):
        kpis, report = run_main(tmp_path, [RENT[0], ing("c", "Client", "500.00", "CREDIT")],
                                [RENT[1], cat("Client", -500.0, "Income"), cat("Client", -500.0, "Income")])
        assert kpis["total_income"] == 500.0
        assert report["categorized_duplicate"] == 1


# ------------------------------------------------------ end-to-end (main())

class TestMainEndToEnd:
    def test_main_writes_kpis_and_join_report(self, tmp_path):
        (tmp_path / "data_quality.json").write_text(json.dumps({"counts": {"rows": 2}}))
        kpis, report = run_main(tmp_path, [RENT[0], ing("c", "Client", "500.00", "CREDIT")],
                                [RENT[1], cat("Client", -500.0, "Income")])

        written = json.loads((tmp_path / "kpis.json").read_text())
        assert written["kpis"] == kpis
        assert kpis["net_cash_flow"] == -500.0
        assert report["rows_in_kpis"] == 2
        # ingest's own section of data_quality.json is preserved
        dq = json.loads((tmp_path / "data_quality.json").read_text())
        assert dq["counts"] == {"rows": 2}

    def test_every_row_excluded_gives_zero_kpis_not_an_error(self, tmp_path):
        kpis, report = run_main(tmp_path, [ing("a", "Shop", "10.00", "DEBIT")], [])
        assert report["rows_in_kpis"] == 0
        assert report["join_missing"] == 1
        assert (kpis["total_spend"], kpis["total_income"], kpis["net_cash_flow"]) == (0.0, 0.0, 0.0)
        assert kpis["spend_by_category"] == {}
        assert kpis["top_merchants"] == []

    def test_data_quality_written_even_if_kpi_validation_fails(self, tmp_path, monkeypatch):
        def boom(kpis, transactions):
            raise ValueError("validation failed")
        monkeypatch.setattr(kpis_mod, "validate", boom)
        with pytest.raises(ValueError, match="validation failed"):
            run_main(tmp_path, [RENT[0]], [RENT[1]])
        dq = json.loads((tmp_path / "data_quality.json").read_text())
        assert dq["kpi_join"]["rows_in_kpis"] == 1
        assert not (tmp_path / "kpis.json").exists()

    def test_main_fails_without_ingested_json(self, tmp_path):
        (tmp_path / "categorized.json").write_text(json.dumps({"categorized": [RENT[1]]}))
        with pytest.raises(FileNotFoundError):
            kpis_mod.main(outputs_dir=str(tmp_path))
