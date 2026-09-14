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
values in FullFixtureExpected.
"""
import json
import os

import pytest

import phase3_kpisnoAI as kpis_mod

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "categorized_sample.json")


@pytest.fixture
def full_fixture():
    with open(FIXTURE_PATH) as f:
        return json.load(f)["categorized"]


# Independently computed (via pandas) ground truth for the frozen fixture.
class FullFixtureExpected:
    total_spend = 8979.42
    total_income = 14700.0
    average_expense = 204.08
    top_merchants = ["Square Payroll", "Best Buy", "WeWork"]
    net_cash_flow = 5720.58
    status = "surplus"
    spend_by_category = {
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


# ------------------------------------------------------------ load_transactions

class TestLoadTransactions:
    def test_loads_fixture(self):
        transactions = kpis_mod.load_transactions(FIXTURE_PATH)
        assert len(transactions) == 50
        assert set(transactions[0].keys()) >= {"date", "merchant", "amount", "category"}


# -------------------------------------------------------- split_income_expense

class TestSplitIncomeExpense:
    def test_splits_purely_on_sign(self):
        transactions = [
            {"amount": -100, "category": "Shopping"},  # sign wins over category
            {"amount": 50, "category": "Income"},
            {"amount": -1, "category": "Other"},
        ]
        income, expense = kpis_mod.split_income_expense(transactions)
        assert income == [transactions[0], transactions[2]]
        assert expense == [transactions[1]]

    def test_string_amounts_are_coerced(self):
        transactions = [{"amount": "-5.0"}, {"amount": "5.0"}]
        income, expense = kpis_mod.split_income_expense(transactions)
        assert len(income) == 1
        assert len(expense) == 1


# ----------------------------------------------------------- compute_core_kpis

class TestComputeCoreKpis:
    def test_full_fixture_matches_independent_calculation(self, full_fixture):
        result = kpis_mod.compute_core_kpis(full_fixture)
        assert result["total_spend"] == FullFixtureExpected.total_spend
        assert result["total_income"] == FullFixtureExpected.total_income
        assert result["average_expense"] == FullFixtureExpected.average_expense
        assert result["top_merchants"] == FullFixtureExpected.top_merchants

    def test_top_merchants_ranks_by_summed_spend_not_transaction_count(self):
        transactions = [
            {"merchant": "A", "amount": 10},
            {"merchant": "A", "amount": 10},
            {"merchant": "A", "amount": 10},  # A totals 30 across 3 small txns
            {"merchant": "B", "amount": 40},  # B totals 40 in a single txn
        ]
        result = kpis_mod.compute_core_kpis(transactions)
        assert result["top_merchants"][0] == "B"

    def test_no_expenses_gives_zero_average_and_empty_merchants(self):
        transactions = [{"merchant": "X", "amount": -100}]
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
        transactions = [{"amount": -10}, {"amount": 100}]
        result = kpis_mod.compute_net_cash_flow(transactions)
        assert result["net_cash_flow"] == -90.0
        assert result["status"] == "deficit"

    def test_zero_net_counts_as_surplus(self):
        transactions = [{"amount": -50}, {"amount": 50}]
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
        transactions = [{"amount": -100, "category": "Income"}]
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
        transactions = [{"merchant": "A", "amount": 100}]
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
        transactions = [
            {"amount": 50, "category": "Dining"},
            {"amount": 25, "category": "Shopping"},
        ]
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
        transactions = [
            {"amount": 100, "date": "10-01-2024"},
            {"amount": 50, "date": "10-01-2024"},
        ]
        result = kpis_mod.compute_burn_rate(transactions)
        assert result["period_days"] == 1
        assert result["daily_avg_spend"] == 150.0

    def test_fewer_than_two_valid_dates_falls_back_to_one_day(self):
        transactions = [
            {"amount": 100, "date": "not-a-date"},
            {"amount": 50, "date": "also-bad"},
        ]
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

    def test_raises_on_nonpositive_total_spend(self):
        with pytest.raises(ValueError, match="total_spend"):
            kpis_mod.validate({"total_spend": 0, "spend_by_category": {}}, [])

    def test_raises_when_category_percentages_dont_sum_to_100(self):
        bad_kpis = {
            "total_spend": 100,
            "spend_by_category": {
                "Shopping": {"amount": 100, "pct_of_spend": 50.0},  # should be ~100
            },
        }
        with pytest.raises(ValueError, match="category pct sum off"):
            kpis_mod.validate(bad_kpis, [])


# ------------------------------------------------------ end-to-end (main())

class TestMainEndToEnd:
    def test_main_writes_valid_kpis_json_for_the_real_csv_derived_fixture(self, tmp_path, monkeypatch, full_fixture):
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        (outputs_dir / "categorized.json").write_text(json.dumps({"categorized": full_fixture}))

        monkeypatch.chdir(tmp_path)
        returned_kpis = kpis_mod.main()

        written = json.loads((outputs_dir / "kpis.json").read_text())
        assert written["kpis"] == returned_kpis
        assert written["kpis"]["total_spend"] == FullFixtureExpected.total_spend
        assert written["kpis"]["burn_rate"]["unparseable_dates"] == FullFixtureExpected.unparseable_dates
