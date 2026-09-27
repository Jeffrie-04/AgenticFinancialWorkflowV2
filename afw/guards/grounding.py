"""
afw/guards/grounding.py — every number in narrative text must be a KPI value.

The summary and advisor text are written by a model, and owners read them.
The model may not do math: each number it writes must be a value in
kpis.json, at the precision the text shows. A correct but derived number
("the remaining 7.8%") still fails.

How numbers are found:
- KPI names (merchants, the top client, categories) are blanked first, so
  digits inside a name ("7-Eleven") aren't numbers; so are list markers at
  the start of a line ("1.", "2)").
- Each digit number is typed by its context: money ("$1,234" or
  "1,234 dollars"), percent ("51.1%", "51.1 percent", "51.1 per cent"), or
  bare. Number words ("ten") and abbreviations ("$48.1k") aren't parsed, so
  an abbreviation fails as the bare number it starts with.

How they match: a number with d decimals matches a KPI of a compatible type
whose magnitude, rounded half-up to d decimals, is equal. Percent matches
percent KPIs, money matches money KPIs, and a bare number matches money or
count KPIs, or, if it's an integer, a structural integer the KPIs are built
on (3 for "top 3", 30 for the monthly projection, and the number of top
merchants and of spending categories).

This module is pure: no I/O and no model calls.
"""
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

COUNT_KEYS = {"period_days", "num_income_sources", "unparseable_dates"}
STRUCTURAL_INTEGERS = {3, 30}

NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
TOKEN = re.compile(
    rf"(?P<dollar>\$ ?)?(?P<num>{NUMBER})(?P<unit> ?%| ?percent\b| ?per cent\b| dollars\b)?",
    re.IGNORECASE)
LIST_MARKER = re.compile(r"(?m)^[ \t]*\d+[.)][ \t]")

FALLBACK_SUMMARY_MARKER = ("[Auto-generated summary: the model's text contained numbers not found in the "
                           "KPIs, so this was built from the KPIs directly.]")
FALLBACK_REFLECTION_MARKER = ("[Auto-generated: the advisor text contained numbers not found in the KPIs, so "
                              "no advice is shown for this period. Re-run the pipeline for the advisor's text.]")


@dataclass
class Grounding:
    ok: bool
    unsupported: list = field(default_factory=list)  # normalized tokens, e.g. "8.0%", "$746", "2"


def _kind(key):
    """A KPI number's type, from its key: the one rule shared by display and matching."""
    return "percent" if key and "pct" in key else "count" if key in COUNT_KEYS else "money"


def display_kpis(value, key=None):
    """The KPIs as the narrative prompts show them: money as $58,860.00 (or
    -$2,204.05), percentages as 51.1%, counts as plain integers; names and
    other strings unchanged. Every displayed form passes check_grounding."""
    if isinstance(value, dict):
        return {k: display_kpis(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [display_kpis(v, key) for v in value]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    kind = _kind(key)
    if kind == "percent":
        return f"{value:.1f}%"
    if kind == "count":
        return int(value)
    return f"-${abs(value):,.2f}" if value < 0 else f"${value:,.2f}"


def _kpi_values(kpis):
    """Magnitudes of every number in the KPIs, grouped by type."""
    values = {"percent": set(), "money": set(), "count": set()}

    def walk(obj, key):
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            values[_kind(key)].add(abs(Decimal(str(obj))))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, k)
        elif isinstance(obj, list):
            for v in obj:
                walk(v, key)

    walk(kpis, None)
    structural = STRUCTURAL_INTEGERS | {len(kpis.get("top_merchants", [])), len(kpis.get("spend_by_category", {}))}
    values["structural"] = {Decimal(n) for n in structural}
    return values


def _kpi_names(kpis):
    """Every string in the KPIs (names, categories), longest first."""
    names = set()

    def walk(obj):
        if isinstance(obj, str):
            names.add(obj)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):  # spend_by_category: category names are keys
                    names.add(k)
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(kpis)
    return sorted((n for n in names if n), key=len, reverse=True)


def _matches(value, decimals, candidates):
    step = Decimal(1).scaleb(-decimals)
    return any(c.quantize(step, rounding=ROUND_HALF_UP) == value for c in candidates)


def check_grounding(text, kpis):
    values = _kpi_values(kpis)
    text = LIST_MARKER.sub(" ", text)
    for name in _kpi_names(kpis):
        text = text.replace(name, " ")

    unsupported = []
    for m in TOKEN.finditer(text):
        num = m.group("num")
        unit = (m.group("unit") or "").strip().lower()
        value = Decimal(num.replace(",", ""))
        decimals = len(num.split(".")[1]) if "." in num else 0
        if unit in ("%", "percent", "per cent"):
            token, candidates = f"{num}%", values["percent"]
        elif m.group("dollar") or unit == "dollars":
            token, candidates = f"${num}", values["money"]
        else:
            token = num
            candidates = values["money"] | values["count"] | (values["structural"] if decimals == 0 else set())
        if not _matches(value, decimals, candidates) and token not in unsupported:
            unsupported.append(token)
    return Grounding(ok=not unsupported, unsupported=unsupported)


# ------------------------------------------------------------------ fallbacks


def _money(value):
    return f"${abs(value):,.2f}"


def _pct(value):
    return f"{value:.1f}%"


def fallback_summary(kpis):
    """A deterministic summary built only from KPI values, marked as
    auto-generated. Grounded by construction (see the tests)."""
    parts = [FALLBACK_SUMMARY_MARKER,
             (f"Total income was {_money(kpis['total_income'])} and total spend was "
              f"{_money(kpis['total_spend'])}, a net cash flow {kpis['status']} of "
              f"{_money(kpis['net_cash_flow'])}.")]
    categories = kpis.get("spend_by_category") or {}
    if categories:
        name, largest = next(iter(categories.items()))
        parts.append(f"The largest spending category was {name} at {_money(largest['amount'])} "
                     f"({_pct(largest['pct_of_spend'])} of spend).")
    income = kpis.get("income_concentration") or {}
    if income.get("top_client"):
        sources = income["num_income_sources"]
        parts.append(f"The top client, {income['top_client']}, accounted for {_pct(income['top_client_pct'])} "
                     f"of income, and the top clients together {_pct(income['top3_clients_pct'])}, across "
                     f"{sources} income source{'' if sources == 1 else 's'}.")
    return " ".join(parts)


def fallback_reflection(kpis):
    """Key figures only, no advice: advice can't be generated deterministically."""
    figures = [f"net cash flow {kpis['status']} of {_money(kpis['net_cash_flow'])}",
               f"fixed costs {_pct(kpis['fixed_vs_discretionary']['fixed_pct'])} of spend"]
    income = kpis.get("income_concentration") or {}
    if income.get("top_client"):
        figures.append(f"top client {_pct(income['top_client_pct'])} of income")
    return f"{FALLBACK_REFLECTION_MARKER} Key figures: {'; '.join(figures)}."
