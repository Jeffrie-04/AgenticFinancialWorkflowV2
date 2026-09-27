# Categorizer eval: v1 vs v2

- Gold set: `eval/gold_set.csv`, sha256 `588958b144ab57dce3014f392aad28d8deee4d2e511112e10fe773f9359d0789`, 142 labeled purchases (0 unlabeled, skipped).
- Model: anthropic_direct / claude-haiku-4-5 @ default (v1), anthropic_direct / claude-haiku-4-5 @ default (v2).
- Results are a single run, pinned by cache (`eval/cache/`): re-running costs nothing and reproduces these numbers; a fresh run against the model could differ. v1 generated 2026-09-27, v2 generated 2026-09-27.
- Scoring: v1 can't answer Travel/Transportation, so gold Travel/Transportation labels count as Other when scoring v1. Review rows count as incorrect.

| Metric | v1 | v2 |
|---|---|---|
| Overall accuracy | 95.1% | 95.8% |
| Accuracy on answered rows | 95.1% | 95.8% |
| Review rows | 0 | 0 |
| Travel/Transportation precision | — | 100.0% |
| Travel/Transportation recall | — | 97.3% |
| Travel/Transportation F1 | — | 98.6% |

## Per-category accuracy (recall on gold labels)

| Category | v1 | v2 |
|---|---|---|
| Utilities | 95.9% (47/49) | 95.9% (47/49) |
| Shopping | 91.7% (33/36) | 91.7% (33/36) |
| Dining | 100.0% (12/12) | 100.0% (12/12) |
| Travel/Transportation | — | 97.3% (36/37) |
| Other | 95.6% (43/45) | 100.0% (8/8) |

## With the original labels

The same model replies, scored against the gold set as first labeled:

- Gold set: `eval/gold_set.csv`, sha256 `b35386c6a4c3528f18cc74ecc72d26b6e0099c65a95abed1494bb471371c608d`, 142 labeled purchases (0 unlabeled, skipped).
- Model: anthropic_direct / claude-haiku-4-5 @ default (v1), anthropic_direct / claude-haiku-4-5 @ default (v2).
- Results are a single run, pinned by cache (`eval/cache/`): re-running costs nothing and reproduces these numbers; a fresh run against the model could differ. v1 generated 2026-09-27, v2 generated 2026-09-27.
- Scoring: v1 can't answer Travel/Transportation, so gold Travel/Transportation labels count as Other when scoring v1. Review rows count as incorrect.

| Metric | v1 | v2 |
|---|---|---|
| Overall accuracy | 81.7% | 82.4% |
| Accuracy on answered rows | 81.7% | 82.4% |
| Review rows | 0 | 0 |
| Travel/Transportation precision | — | 100.0% |
| Travel/Transportation recall | — | 97.3% |
| Travel/Transportation F1 | — | 98.6% |

### Per-category accuracy (recall on gold labels)

| Category | v1 | v2 |
|---|---|---|
| Utilities | 83.3% (40/48) | 83.3% (40/48) |
| Shopping | 88.5% (23/26) | 88.5% (23/26) |
| Dining | 66.7% (12/18) | 66.7% (12/18) |
| Travel/Transportation | — | 97.3% (36/37) |
| Other | 82.0% (41/50) | 46.2% (6/13) |

## Label audit

The first scoring (the "original labels" results) showed misses where the gold label contradicted both prompts' own guide lines. That prompted a label audit. The correction was then applied by rule to every gold row in four classes, not only to the rows the model missed:

- **Payroll → Utilities** (the Utilities guide line lists "payroll"). Changed: rows 17, 18, 75, 76 (Gusto Payroll). Already Utilities: rows 33, 34 (ADP Payroll).
- **Professional services → Utilities** (the Utilities guide line lists "professional services"). Changed: rows 46 (Certified Court Reporting), 47 (Expert Witness Retainer), 141 (Brightline Consulting Partners). Already Utilities: rows 35 (Contract Attorney Payment), 48 (Process Server LLC).
- **Food and ingredient suppliers → Shopping** (the Shopping guide line covers "retail and supplies"). Changed: rows 63–66 (Sysco Foods), 67–68 (US Foods), 71–72 (Local Seafood Co), 73–74 (Craft Beverage Distributor). Already Shopping: rows 69–70 (Restaurant Depot).
- **Waste services → Other** (both prompts' Other guide line lists "waste services"). Changed: row 22 (County Landfill, yard waste disposal), first as a consistency fix to match row 23 (County Landfill, debris disposal), an identical transaction type; then row 83 (Waste Management, commercial trash and grease disposal), in a follow-up applying the same class. Already Other: row 23.

Deliberately unchanged: row 61 (LegalZoom Filing) stays Other. It's a filing fee, consistent with row 58 (County Clerk Filing Fee).

19 labels changed in total (row numbers count data rows, header excluded). No prompt text changed. Both versions were re-scored from the cached model replies with 0 model calls, so they're compared on the same corrected labels. Because the audit followed the first results, the corrected scores aren't a blind measurement; the original-label scores above are kept for that reason.
