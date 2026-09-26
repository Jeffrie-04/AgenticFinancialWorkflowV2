# Frozen KPI snapshots

Each folder holds a business's `transactions.csv`, `categorized.json` and
`kpis.json`, frozen at commit `45b03e2`. `test_kpi_snapshot.py` checks that the
current pipeline still produces exactly this `kpis.json`, and `git diff 45b03e2
-- 'tests/fixtures/snapshots/**/kpis.json'` must stay empty.

The `categorized.json` files are **frozen v1-era model output**, converted to
`{id, category}` by `scripts/convert_snapshot_categorized.py`. They still have
entries for CREDIT rows, which the categorizer no longer sends to the model
(CREDITs are Income by rule), and they have no `review` or `llm` keys. The KPI
join accepts them as they are, so don't regenerate them to match the current
categorizer output.
