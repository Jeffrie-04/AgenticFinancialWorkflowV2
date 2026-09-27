# Categorizer eval: v1 vs v2

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

## Per-category accuracy (recall on gold labels)

| Category | v1 | v2 |
|---|---|---|
| Utilities | 83.3% (40/48) | 83.3% (40/48) |
| Shopping | 88.5% (23/26) | 88.5% (23/26) |
| Dining | 66.7% (12/18) | 66.7% (12/18) |
| Travel/Transportation | — | 97.3% (36/37) |
| Other | 82.0% (41/50) | 46.2% (6/13) |
