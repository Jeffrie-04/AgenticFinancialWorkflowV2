"""
Snapshot test: the KPI numbers for the three demo businesses must not change.

tests/fixtures/snapshots/<business>/ holds frozen copies of that business's
transactions.csv, outputs/categorized.json and outputs/kpis.json, taken
before the ingest/KPI refactor. The expected kpis.json files are never edited:
if a refactor moves any figure, this test fails. Using frozen copies means
re-running the pipeline (which rewrites businesses/*/outputs/) can't break it.

The KPIs now come from the validated source CSV (afw/ingest.py) with only the
category taken from the LLM's categorized.json — and still equal the frozen
numbers computed the old way.
"""
import json
import os
import shutil

import pytest

import phase3_kpisnoAI as kpis_mod
from afw import ingest

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "snapshots")
BUSINESSES = ["landscaper", "law_firm", "restaurant"]
CSV_ROWS = {"landscaper": 42, "law_firm": 35, "restaurant": 47}


@pytest.mark.parametrize("business", BUSINESSES)
def test_kpis_match_frozen_snapshot(business, tmp_path):
    src = os.path.join(SNAPSHOT_DIR, business)
    ingest.main(csv_path=os.path.join(src, "transactions.csv"), outputs_dir=str(tmp_path))
    shutil.copy(os.path.join(src, "categorized.json"), tmp_path / "categorized.json")

    kpis_mod.main(outputs_dir=str(tmp_path))

    with open(os.path.join(src, "kpis.json")) as f:
        expected = json.load(f)
    actual = json.loads((tmp_path / "kpis.json").read_text())
    assert actual == expected

    dq = json.loads((tmp_path / "data_quality.json").read_text())
    rows = CSV_ROWS[business]
    assert dq["counts"] == {"rows": rows, "ok": rows, "needs_review": 0, "rejected": 0}
    join = dq["kpi_join"]
    assert join["rows_in_kpis"] == rows
    assert join["excluded_total"] == {"debit": "0.00", "credit": "0.00", "rows": 0}
    assert all(join[k] == 0 for k in kpis_mod.JOIN_PROBLEMS)
