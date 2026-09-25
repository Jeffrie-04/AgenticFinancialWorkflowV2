"""
Snapshot test: the KPI numbers for the three demo businesses must not change.

tests/fixtures/snapshots/<business>/ holds frozen copies of that business's
transactions.csv, outputs/categorized.json and outputs/kpis.json, taken
before the ingest/KPI refactor. The expected kpis.json files are never edited:
if a refactor moves any figure, this test fails. Using frozen copies means
re-running the pipeline (which rewrites businesses/*/outputs/) can't break it.
"""
import json
import os
import shutil

import pytest

import phase3_kpisnoAI as kpis_mod

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "snapshots")
BUSINESSES = ["landscaper", "law_firm", "restaurant"]


@pytest.mark.parametrize("business", BUSINESSES)
def test_kpis_match_frozen_snapshot(business, tmp_path):
    src = os.path.join(SNAPSHOT_DIR, business)
    shutil.copy(os.path.join(src, "categorized.json"), tmp_path / "categorized.json")

    kpis_mod.main(outputs_dir=str(tmp_path))

    with open(os.path.join(src, "kpis.json")) as f:
        expected = json.load(f)
    actual = json.loads((tmp_path / "kpis.json").read_text())
    assert actual == expected
