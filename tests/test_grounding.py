"""
Tests for afw/guards/grounding.py — every number in the summary and advisor
text must be a KPI value at the precision shown in the text.

tests/fixtures/grounding/ holds the narrative text committed with the demo
outputs; their kpis.json files are identical to tests/fixtures/snapshots/,
which is what these tests check them against.
"""
import json
import os

import pytest

import phase3_kpisnoAI as kpis_mod
from afw.guards.grounding import (
    FALLBACK_REFLECTION_MARKER,
    FALLBACK_SUMMARY_MARKER,
    check_grounding,
    fallback_reflection,
    fallback_summary,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
BUSINESSES = ["landscaper", "law_firm", "restaurant"]


def kpis_of(business):
    with open(os.path.join(FIXTURES, "snapshots", business, "kpis.json")) as f:
        return json.load(f)["kpis"]


def text_of(name):
    with open(os.path.join(FIXTURES, "grounding", name), encoding="utf-8") as f:
        return f.read()


LANDSCAPER = kpis_of("landscaper")

# --------------------------------------------------- the committed narratives


def test_real_summary_with_derived_percentage_fails():
    # "The remaining 8.0%" is Other 7.0 + Dining 0.8 = 7.8, computed (wrongly) by the model.
    result = check_grounding(text_of("landscaper_summary.txt"), LANDSCAPER)
    assert not result.ok
    assert result.unsupported == ["8.0%"]


def test_correct_summary_passes():
    text = text_of("landscaper_summary.txt").replace(
        "The remaining 8.0% was split between Other and Dining.",
        "Other took 7.0% and Dining 0.8%.")
    result = check_grounding(text, LANDSCAPER)
    assert result.ok and result.unsupported == []


def test_reflection_forms_without_dollar_sign_and_percent_word_pass():
    # "58,860", "48,147", "10,713", "48.6 percent", "24,609", "23,538": all KPIs at 0 or 1 dp.
    assert check_grounding(text_of("landscaper_reflection.txt"), LANDSCAPER).ok


def test_derived_sum_fails():
    result = check_grounding(text_of("law_firm_summary.txt"), kpis_of("law_firm"))
    assert result.unsupported == ["$746"]  # Dining + Other, not a KPI


def test_rounded_numbers_and_advice_numbers_fail():
    result = check_grounding(text_of("law_firm_reflection.txt"), kpis_of("law_firm"))
    # "nearly 80,000" (79,712.98) and "2-3 new recurring clients"; 3 is structural (top 3)
    assert result.unsupported == ["80,000", "2"]


def test_approximations_in_words_fail():
    result = check_grounding(text_of("restaurant_reflection.txt"), kpis_of("restaurant"))
    # "about 2,200 dollars" (2,204.05), "over 80 percent" (80.9), "34,000 dollars" (34,080.50)
    assert result.unsupported == ["$2,200", "80%", "$34,000"]  # tokens are reported normalized


# ------------------------------------------------------------ matching rules


@pytest.mark.parametrize("text", [
    "Income was $58,860.00.", "Income was $58,860.", "Income was 58,860 dollars.", "Income was 58860.",
    "Utilities were 51.1% of spend.", "Utilities were 51.1 percent of spend.", "Utilities were 51 per cent.",
    "Over a 28-day period.", "The top 3 clients.", "A monthly projection over 30 days.",
    "A surplus of $10,713.", "Utilities were 51.10% of spend.",  # 51.1 at 2 dp is 51.10
])
def test_numbers_match_at_the_precision_shown(text):
    assert check_grounding(text, LANDSCAPER).ok, check_grounding(text, LANDSCAPER).unsupported


@pytest.mark.parametrize("text,bad", [
    ("Income was $58,860.01.", "$58,860.01"),  # not the KPI at the precision shown
    ("Utilities were 51.2% of spend.", "51.2%"),
    ("Income was $48.1k.", "$48.1"),            # abbreviations are not parsed
    ("Spend grew 51.1 dollars.", "$51.1"),      # money must match a money KPI
    ("There were 2 new clients.", "2"),         # not a KPI, not structural
    ("In October 2024.", "2024"),               # dates aren't KPIs
])
def test_unsupported_numbers(text, bad):
    assert check_grounding(text, LANDSCAPER).unsupported == [bad]


def test_rounding_is_half_up():
    kpis = {"total_spend": 24608.5, "spend_by_category": {}, "top_merchants": []}
    assert check_grounding("Spend was $24,609.", kpis).ok
    assert check_grounding("Spend was $24,608.", kpis).unsupported == ["$24,608"]


def test_percent_must_match_a_percent_kpi():
    # 28 is a KPI (period_days) but not a percentage.
    assert check_grounding("Margins were 28%.", LANDSCAPER).unsupported == ["28%"]


def test_money_must_match_a_money_kpi():
    # 10 is a KPI (num_income_sources) but not a dollar amount.
    assert check_grounding("It cost $10.", LANDSCAPER).unsupported == ["$10"]


def test_negative_values_match_by_magnitude():
    kpis = kpis_of("restaurant")
    assert kpis["net_cash_flow"] == -2204.05
    assert check_grounding("A net cash flow deficit of $2,204.05 (-$2,204.05).", kpis).ok


def test_digits_inside_kpi_names_are_not_numbers():
    kpis = {**LANDSCAPER, "top_merchants": ["7-Eleven", "Route 66 Diner", "Shell Gas"]}
    assert check_grounding("Top merchants were 7-Eleven and Route 66 Diner.", kpis).ok


def test_list_markers_at_line_start_are_ignored():
    text = "1. Income was $58,860.00.\n2) Spend was $48,147.25.\n  3. Net was $10,712.75."
    assert check_grounding(text, LANDSCAPER).ok


def test_text_without_numbers_passes():
    assert check_grounding("Your business is healthy.", LANDSCAPER).ok


# --------------------------------------------------------------- fallbacks


def kpis_from(transactions):
    """KPIs computed by the real engine for hand-made rows (edge cases)."""
    t = kpis_mod.apply_refunds(transactions)
    kpis = {}
    kpis.update(kpis_mod.compute_core_kpis(t))
    kpis.update(kpis_mod.compute_net_cash_flow(t))
    kpis["spend_by_category"] = kpis_mod.compute_spend_by_category(t)
    kpis["income_concentration"] = kpis_mod.compute_income_concentration(t)
    kpis["fixed_vs_discretionary"] = kpis_mod.compute_fixed_vs_discretionary(t)
    kpis["burn_rate"] = kpis_mod.compute_burn_rate(t)
    return kpis


def row(merchant, amount, direction, category="Other", day="2024-10-01"):
    return {"id": merchant, "merchant": merchant, "amount": amount, "direction": direction,
            "category": category, "date": day, "is_refund": False}


EDGE_KPIS = {
    "deficit": kpis_from([row("Client", "100.00", "CREDIT", "Income"), row("Rent", "900.00", "DEBIT", "Utilities")]),
    "zero_spend": kpis_from([row("Client", "100.00", "CREDIT", "Income")]),
    "no_income": kpis_from([row("Rent", "900.00", "DEBIT", "Utilities"), row("Cafe", "12.34", "DEBIT", "Dining")]),
    "nothing": kpis_from([]),
    "digit_names": kpis_from([row("Studio 54 LLC", "5400.00", "CREDIT", "Income"),
                              row("7-Eleven", "7.11", "DEBIT", "Dining")]),
}
ALL_KPIS = {**{b: kpis_of(b) for b in BUSINESSES}, **EDGE_KPIS}


@pytest.mark.parametrize("name", list(ALL_KPIS))
def test_fallbacks_always_pass_grounding(name):
    kpis = ALL_KPIS[name]
    for text in (fallback_summary(kpis), fallback_reflection(kpis)):
        result = check_grounding(text, kpis)
        assert result.ok, (name, text, result.unsupported)


def test_fallback_summary_content_and_marker():
    text = fallback_summary(LANDSCAPER)
    assert text.startswith(FALLBACK_SUMMARY_MARKER)
    for part in ("$58,860.00", "$48,147.25", "surplus of $10,712.75", "Utilities at $24,609.00 (51.1% of spend)",
                 "Greenfield Estate Project", "21.2%", "48.6%", "10 income sources"):
        assert part in text


def test_fallback_reflection_content_and_marker():
    text = fallback_reflection(kpis_of("restaurant"))
    assert text.startswith(FALLBACK_REFLECTION_MARKER)
    assert "deficit of $2,204.05" in text and "45.2% of spend" in text and "80.9% of income" in text


def test_fallbacks_omit_sentences_without_data():
    nothing = EDGE_KPIS["nothing"]
    assert "largest spending category" not in fallback_summary(nothing)
    assert "top client" not in fallback_summary(nothing)
    assert "top client" not in fallback_reflection(nothing)
