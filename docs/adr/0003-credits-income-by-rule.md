# ADR 0003: Credits are Income by rule, not by the model

- **Status:** Accepted
- **Date:** 2026-09-26
- **Phase:** 2 (treat the model as untrusted)

## Context

The categorizer prompt already told the model that every credit is Income. It
said "Every CREDIT row is Income" in v1.1, and before that "NEGATIVE amounts
are ALWAYS Income". For credits, the model's answer was only a restatement of
a rule we had written ourselves. Sending those rows anyway had three costs:

- **Cost.** Tokens were spent on rows whose answer was already known.
- **Risk.** Once replies are validated, a credit the model labels anything
  other than Income has to become NEEDS_REVIEW, and NEEDS_REVIEW rows are left
  out of the KPIs. A wrong label on a rule-determined row would silently
  remove real income from the report.
- **Client names leaving the machine.** Credits carry client names, which are
  often people or households ("Patterson Residence Install"). Every credit
  sent to the model sent a client name to a third-party API.

For income, the category is also cosmetic: the KPIs split income from spend by
`direction`, not by category (ADR 0001).

## Options

1. **Send all rows, strict rule.** A non-refund CREDIT must come back as
   Income; anything else is NEEDS_REVIEW. This keeps every cost above, and a
   wrong label drops income from the KPIs.
2. **Send all rows, lenient rule.** A non-refund CREDIT may be Income or
   Other. There are fewer false reviews, but the label is still decided by the
   model, and client names are still sent.
3. **Assign by rule.** Credits are never sent. Each non-refund CREDIT is
   Income, deterministically.

## Decision

Option 3.

- Only non-refund DEBITs go to the model (`afw/llm_input.model_rows`). The
  categorizer prompt lists only the DEBIT categories: Utilities, Shopping,
  Dining and Other.
- Every non-refund CREDIT is written to `categorized.json` as Income by the
  categorizer itself, and counted in `llm.rule_assigned`.
- Refunds aren't sent either. A refund keeps the category of the original
  DEBIT it is netted against (the refund rule from Phase 1), and is counted in
  `llm.refunds_not_sent`.
- The direction check reduces to "a DEBIT can't be Income". It's enforced
  when replies are validated, and again at the KPI join, so a stale or
  hand-edited `categorized.json` can't bypass it.
- The plan phase's sample rows follow the same rule, so they contain
  non-refund DEBITs only.

## Consequences

**Positive**

- The model can't remove income from the KPIs, and fewer tokens are spent.
- Client names no longer reach the categorizer or plan prompts. They still
  reach the summary and reflection prompts through `top_client`, by the
  deliberate choice recorded in ADR 0002.
- The KPI figures are unchanged. The frozen snapshots prove it, because
  income was already determined by direction.

**Costs**

- **Everything that comes in is labelled Income.** That includes transfers
  in, loan proceeds and owner contributions. There's no transfer detection
  yet (`is_transfer` is never set), so this isn't a new error; the KPIs
  already counted every non-refund CREDIT as income. But the label now states
  it outright.
- **Refunds from an earlier period.** A refund whose original DEBIT isn't in
  the same file is still treated as ordinary income, unless it's flagged as a
  possible refund (NEEDS_REVIEW), as in Phase 1.
- **Old fixtures.** The frozen v1-era `categorized.json` fixtures in
  `tests/fixtures/snapshots/` still contain model answers for CREDIT rows.
  The KPI join accepts them, and they are kept as frozen history.
