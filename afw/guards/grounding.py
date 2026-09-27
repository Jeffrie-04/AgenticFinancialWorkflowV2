"""
afw/guards/grounding.py — every number in narrative text must be a KPI value.

The summary and advisor text are written by a model, and owners read them.
The model may not do math: each number it writes must be a value in
kpis.json, at the precision the text shows. A correct but derived number
("the remaining 7.8%") still fails.

How numbers are found:
- The text is normalized first: Unicode format characters are removed, list
  markers at the start of a line ("1.", "2)") are dropped, and whitespace
  runs collapse to one space.
- KPI names (merchants, the top client, categories) are blanked, but only
  as whole spans: not next to a letter or digit, and never inside a number
  (a merchant named "58" doesn't touch "$58,860.00"; one named "999"
  doesn't hide "$999").
- Each digit number is typed by its context: money ("$1,234" or
  "1,234 dollars"), percent ("51.1%", "51.1 percent", "51.1 per cent"), or
  bare. A "-" or "\u2212" written directly before it is its sign.
- A number directly followed by letters ("$100k", "$58,860thousand",
  "3rd") is unsupported as a whole, as is one with more than MAX_DECIMALS
  decimals. Nothing here raises: an unusable number is just unsupported.
  Number words ("ten") aren't parsed.

How they match: a number with d decimals matches a KPI of a compatible type
that, rounded half-up to d decimals, is equal: by magnitude when no sign
is written ("a deficit of $2,204.05" matches -2,204.05), and with its sign
when one is ("-$58,860.00" doesn't match income of 58,860.00). Percent
matches percent KPIs, money matches money KPIs, and a bare number matches
money or count KPIs, or, if it's an integer, a structural integer the KPIs
are built on (3 for "top 3", 30 for the monthly projection, and the number
of top merchants and of spending categories).

This module is pure: no I/O and no model calls.
"""
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from afw.guards.pii import strip_format_characters

COUNT_KEYS = {"period_days", "num_income_sources", "unparseable_dates"}
STRUCTURAL_INTEGERS = {3, 30}

MAX_DECIMALS = 6  # KPIs have 2; anything past this is unsupported, never computed

NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
SIGNS = "-\u2212"
TOKEN = re.compile(
    rf"(?P<sign>(?<![\w.,])[{SIGNS}])?(?P<dollar>\$ ?)?(?P<num>{NUMBER})"
    rf"(?P<unit> ?%| ?percent\b| ?per cent\b| dollars\b)?",
    re.IGNORECASE)
LETTERS_AFTER = re.compile(r"[^\W\d_]+")
LIST_MARKER = re.compile(r"(?m)^[ \t]*\d+[.)][ \t]")
WHITESPACE_RUN = re.compile(r"\s+")

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
            values[_kind(key)].add(Decimal(str(obj)))  # signed; unsigned text compares magnitudes
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


def _matches(value, decimals, candidates, signed):
    """value: the number as written (negative if a sign was written)."""
    step = Decimal(1).scaleb(-decimals)
    try:
        return any((c if signed else abs(c)).quantize(step, rounding=ROUND_HALF_UP) == value for c in candidates)
    except InvalidOperation:  # unreachable while MAX_DECIMALS bounds `step`; kept as a safety net
        return False


def _normalize(text):
    text = strip_format_characters(text)
    text = LIST_MARKER.sub(" ", text)
    return WHITESPACE_RUN.sub(" ", text)


def _blank_names(text, kpis):
    """Blank each KPI name where it stands as a whole span: not next to a
    letter or digit, and never inside a number ("$58,860.00", "51.1%")."""
    for name in _kpi_names(kpis):
        name = WHITESPACE_RUN.sub(" ", name).strip()
        if name:
            text = re.sub(rf"(?<![\w$.,{SIGNS}]){re.escape(name)}(?![\w%]|[.,]\d)", " ", text)
    return text


def check_grounding(text, kpis):
    values = _kpi_values(kpis)
    text = _blank_names(_normalize(text), kpis)

    unsupported = []
    for m in TOKEN.finditer(text):
        sign, num = m.group("sign") or "", m.group("num")
        unit = (m.group("unit") or "").strip().lower()
        if unit in ("%", "percent", "per cent"):
            token, candidates = f"{sign}{num}%", values["percent"]
        elif m.group("dollar") or unit == "dollars":
            token, candidates = f"{sign}${num}", values["money"]
        else:
            token, candidates = f"{sign}{num}", values["money"] | values["count"]
        decimals = len(num.split(".")[1]) if "." in num else 0
        tail = None if unit else LETTERS_AFTER.match(text, m.end())

        if tail:  # "$100k", "$58,860thousand", "3rd": unsupported as a whole
            token, ok = token + tail.group(), False
        elif decimals > MAX_DECIMALS:
            ok = False
        else:
            value = Decimal(num.replace(",", "")) * (-1 if sign else 1)
            if not unit and not m.group("dollar") and not sign and decimals == 0:
                candidates = candidates | values["structural"]
            ok = _matches(value, decimals, candidates, signed=bool(sign))
        if not ok and token not in unsupported:
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
