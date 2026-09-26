# ADR 0001: Positive amount plus direction instead of signed amounts

- **Status:** Accepted
- **Date:** 2026-09-24
- **Phase:** 1 (validated ingest), released in v1.1

## Context

Before Phase 1, a transaction's meaning came from the sign of its amount.
Negative meant income and positive meant an expense. That convention lived in
a docstring in `phase3_kpisnoAI.py` and in the categorizer prompt ("NEGATIVE
amounts are ALWAYS Income"), not in the data or in any type.

Banks don't agree on a sign convention. The demo exports follow the
card-statement style, where money in is negative, but a checking-account
export usually has money in as positive. Nothing recorded which convention a
file used, so a source with the other convention would have swapped income
and spend in every KPI without any error. The sign was also read in several
places: the KPI functions, and the LLM, which echoed signed amounts back in
`categorized.json`.

## Options

1. **Keep signed amounts** and document the convention. This needs no
   refactor. But the meaning stays implicit, every consumer has to know the
   convention, and a source with the other convention still silently inverts
   the KPIs.
2. **A positive amount plus a `direction` field (`DEBIT`/`CREDIT`)**,
   converted once at ingest using a per-source `sign_convention`. The meaning
   is explicit in every row, and only ingest knows about bank conventions.
3. **Accept separate debit and credit columns**, which some banks export
   instead of one signed column. This is a real format, but it's a different
   input shape and doesn't remove the need for a direction field downstream.
   Deferred.

## Decision

Option 2.

- `Transaction.amount` is a `Decimal` greater than 0 with exactly 2 decimal
  places. `Transaction.direction` is `DEBIT` (money out) or `CREDIT` (money
  in). Both are enforced by the Pydantic model in `afw/models.py`.
- Each source declares `sign_convention` in its `SourceConfig`, either
  `negative_is_credit` (the default and the demo data's convention) or
  `positive_is_credit`. It is configured per business folder in
  `afw/ingest.py` and is never guessed from the data.
- The conversion happens once, in `afw/ingest.parse_amount`. Everything
  downstream reads `direction` and never the sign: the KPIs split income from
  spend on it, and the categorizer prompt gets a direction instead of an
  amount.
- Refunds are CREDITs linked to their original DEBIT and are netted against
  it, rather than being counted as income.

Option 3 is deferred. A file with separate debit and credit columns is
currently rejected at ingest and listed as a known limitation.

## Consequences

**Positive**

- The meaning of every row is explicit and checked. A row can't be usable
  without a direction.
- Supporting a bank with the other sign convention is a config change, not a
  code change.
- The KPIs no longer depend on the sign or on amounts echoed by the LLM.

**Costs**

- A KPI refactor. The KPI functions now split on `direction`, and the
  existing tests were converted with a small helper. A frozen snapshot of the
  three demo businesses' `kpis.json` proved the figures identical before and
  after.
- Per-source configuration has to be correct. A wrong `sign_convention`
  inverts income and spend for that whole source, and the data alone can't
  detect it. Unknown sources, such as dashboard uploads, get the default.
- Exports with separate debit and credit columns aren't supported yet.

This is not a performance change. Ingest does the same single pass over the
rows. The change is about correctness and making the meaning explicit.
