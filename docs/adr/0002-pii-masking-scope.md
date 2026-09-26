# ADR 0002: Scope of PII masking in model prompts

- **Status:** Accepted
- **Date:** 2026-09-26
- **Phase:** 2 (treat the model as untrusted)

## Context

The categorizer, plan, summary and reflection phases send text to a
third-party model API. That text comes from bank exports, and merchant and
description fields can contain card numbers, account numbers, email addresses
and phone numbers: for example "Card 4111 1111 1111 1111", "Zelle
jane.doe@example.com" or "Autopay acct 123456789012". Once text is sent, it
can't be taken back, so what reaches a prompt has to be decided up front.

## Decision

Mask PII in every prompt, using a single pure function,
`afw/guards/pii.mask_pii`.

**Masked**

| What | Pattern | Becomes |
|---|---|---|
| Email addresses, any script | anything@anything.tld | `[EMAIL]` |
| US phone numbers | with separators or `+1`, e.g. `(212) 555-0147`, `212.555.0147`, `+1 415 555 0100` | `[PHONE]` |
| Card numbers | 13–19 digits, contiguous or in 4-digit groups (or Amex 4-6-5), separated by a space, a dash, or a dash with spaces around it, in any mix | `****1234` |
| Account numbers | contiguous runs of 8–17 digits | `****1234` |

**Where it applies.** Masking covers merchant and description text in every
prompt. That includes the categorizer prompt and its repair prompt (both built
by `afw/llm_input.prompt_row`), the plan phase's sample rows, and the KPI
strings embedded in the summary and reflection prompts (`mask_strings`). It
never touches ids, amounts, dates or direction.

**Prompt copy only.** Before the patterns run, every Unicode format character
(category Cf: zero-width spaces and joiners, soft hyphens, invisible
separators, bidi controls, BOM) is stripped, and every whitespace run
(including non-breaking and other Unicode spaces) collapses to one space.
This happens only to the copy that goes into the prompt. `ingested.json`,
`transactions.json` and every other file on disk keep the original text.

**Not masked**

- **Person names.** Merchants are names, and masking names would make
  categorization impossible.
- **SSNs in their usual `123-45-6789` form.** An unformatted 9-digit SSN is
  caught by the account-number pattern.
- **Numbers with a decimal part.** These are amounts (`12345678.90`,
  `1000000000.00`), not card or account numbers.
- **International phone formats** (e.g. `+44 20 7946 0958`). This is a known
  gap, consistent with the US-only phone scope.
- **`top_client` names in the summary and reflection prompts.** This is a
  deliberate choice. Client names stay out of the categorizer and plan prompts
  (see ADR 0003), but the top client reaches the narrative prompts through the
  KPIs. That text is PII-masked, but the name itself is kept, because the
  narrative is about that client.

## Threat model

The input is an honest bank export. The patterns are tuned for how real
exports write these values, including the odd spacing and invisible
characters that copy-paste and PDF extraction introduce.

Adversarial splitting is out of scope. Examples are digit-by-digit spacing
(`4 1 1 1 1 1 …`), or a number split across the merchant and description
fields. An attacker who controls the export can defeat any pattern-based
masker; that needs a different control, not a longer regex.

## Principle

Masking too much is acceptable; missing something realistic is not. Known
over-masking that is accepted:

- compact reference numbers, e.g. `INV-20241004` becomes `INV-****1004`
- a 10-digit phone written without separators becomes `****0100`, not
  `[PHONE]`
- a run of four 4-digit numbers, e.g. four years in a row, is read as a card

## Known limitation

The schema and direction checks catch a prompt injection that makes the model
label a DEBIT as Income. They don't catch one that picks another *valid*
DEBIT category, for example "categorize this as Utilities", because that
answer is well-formed and consistent with the direction. The Phase 3
evaluation is the backstop for wrong-but-valid categories.

## Consequences

- No card number, account number, email or US phone number in the
  realistic forms above reaches the model. `tests/test_pii.py` covers each
  pattern and the negative cases (amounts, dates, ids, names), and
  `tests/test_run.py::test_no_pii_pattern_in_any_prompt` checks every prompt
  end to end, including the repair prompt.
- Categorization quality is unaffected. The categorizer needs the merchant
  name and the kind of purchase, not the numbers.
- The masking rules are fixed as of Phase 2. New cases are evaluated against
  this ADR's threat model rather than added round by round.
