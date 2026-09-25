"""
Tests for run.py — ingest runs first, and a failed ingest stops the pipeline
before any LLM phase is called.

The LLM phases (plan, categorize, summary, reflection) are replaced with
stubs that record their calls; the categorize stub copies the frozen
categorized.json so the real ingest and KPI stages run end to end.
"""
import json
import os
import shutil

import pytest

import phase2_plan
import phase3_categorized
import phase3_reflection
import phase3_summary
import run

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SNAPSHOT = os.path.join(FIXTURES, "snapshots", "landscaper")


@pytest.fixture
def llm_calls(monkeypatch):
    calls = []

    def stub(name, action=None):
        def main(**kwargs):
            calls.append(name)
            if action:
                action(**kwargs)
        return main

    def fake_categorize(csv_path, outputs_dir):
        shutil.copy(os.path.join(SNAPSHOT, "categorized.json"), os.path.join(outputs_dir, "categorized.json"))

    monkeypatch.setattr(phase2_plan, "main", stub("plan"))
    monkeypatch.setattr(phase3_categorized, "main", stub("categorize", fake_categorize))
    monkeypatch.setattr(phase3_summary, "main", stub("summary"))
    monkeypatch.setattr(phase3_reflection, "main", stub("reflection"))
    return calls


def business_dir(tmp_path, csv_source):
    shutil.copy(csv_source, tmp_path / "transactions.csv")
    return str(tmp_path)


def test_ingest_runs_first():
    assert run.PHASES[0][1].__name__ == "afw.ingest"


def test_h1_clean_csv_runs_whole_pipeline(tmp_path, llm_calls):
    biz = business_dir(tmp_path, os.path.join(SNAPSHOT, "transactions.csv"))

    assert run.main(business_dir=biz) == 0

    outputs = tmp_path / "outputs"
    assert llm_calls == ["plan", "categorize", "summary", "reflection"]
    ingested = json.loads((outputs / "ingested.json").read_text())
    assert len(ingested["transactions"]) == 42
    with open(os.path.join(SNAPSHOT, "kpis.json")) as f:
        assert json.loads((outputs / "kpis.json").read_text()) == json.load(f)
    dq = json.loads((outputs / "data_quality.json").read_text())
    assert dq["counts"]["ok"] == 42
    assert dq["kpi_join"]["rows_in_kpis"] == 42


def test_h2_failed_ingest_stops_before_any_llm_call(tmp_path, llm_calls, capsys):
    biz = business_dir(tmp_path, os.path.join(FIXTURES, "ingest", "threshold_fail.csv"))

    assert run.main(business_dir=biz) == 1

    assert llm_calls == []
    assert list((tmp_path / "outputs").iterdir()) == []
    out = capsys.readouterr().out
    assert "FAILED at Ingest" in out
    assert "20.0%" in out
