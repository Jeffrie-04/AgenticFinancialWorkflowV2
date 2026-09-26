"""
Tests for run.py — ingest runs first, and a failed ingest stops the pipeline
before any LLM phase is called.

The LLM phases (plan, categorize, summary, reflection) are replaced with
stubs that record their calls; the categorize stub copies the frozen
categorized.json so the real ingest and KPI stages run end to end.
"""
import json
import os
import re
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

    def fake_categorize(outputs_dir):
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


def stub_model_calls(monkeypatch):
    """Replace call_model in every phase with a stub that records each prompt
    and returns a valid reply; the real client fails the test if reached."""
    prompts = []

    def fake_call_model(prompt):
        prompts.append(prompt)
        if phase3_categorized.ROWS_START in prompt:
            block = prompt.split(phase3_categorized.ROWS_START, 1)[1].split(phase3_categorized.ROWS_END, 1)[0]
            rows = [json.loads(line) for line in block.strip().splitlines()]
            return json.dumps({"categorized": [
                {"id": r["id"], "category": "Income" if r["direction"] == "CREDIT" else "Other"}
                for r in rows]})
        if "plan_steps" in prompt:
            return json.dumps({"plan_steps": ["plan", "act", "observe", "summarize", "reflect"]})
        return "Stub narrative."

    def no_network(prompt):
        raise AssertionError("a phase called the real model")

    import bedrock_client
    monkeypatch.setattr(bedrock_client, "call_model", no_network)
    for module in (phase2_plan, phase3_categorized, phase3_summary, phase3_reflection):
        monkeypatch.setattr(module, "call_model", fake_call_model)
    return prompts


def test_codex_finding1_only_ok_rows_reach_any_prompt(tmp_path, monkeypatch):
    """Real phase code end to end; only call_model is stubbed. 10 rows:
    8 OK, 1 REJECTED (amount abc), 1 NEEDS_REVIEW (blank merchant) -> 10%
    rejected, so ingest passes and every LLM phase actually builds a prompt."""
    prompts = stub_model_calls(monkeypatch)

    biz = business_dir(tmp_path, os.path.join(FIXTURES, "pipeline", "codex_finding1.csv"))
    assert run.main(business_dir=biz) == 0

    outputs = tmp_path / "outputs"
    dq = json.loads((outputs / "data_quality.json").read_text())
    # The threshold passed, so the absence checks below are meaningful.
    assert dq["counts"] == {"rows": 10, "ok": 8, "needs_review": 1, "rejected": 1}
    assert (outputs / "kpis.json").exists()

    ingested = json.loads((outputs / "ingested.json").read_text())["transactions"]
    ok_ids = [t["id"] for t in ingested if t["status"] == "OK"]
    excluded_ids = [t["id"] for t in ingested if t["status"] != "OK"]
    assert len(ok_ids) == 8 and len(excluded_ids) == 2

    assert len(prompts) == 4  # plan, categorize, summary, reflection
    for prompt in prompts:
        assert "REJECTED_MARKER" not in prompt
        assert "REVIEW_MARKER" not in prompt
        assert not re.search(r"\babc\b", prompt)  # word match: hex ids may contain "abc"
        assert not any(i in prompt for i in excluded_ids)

    # Positive control: the categorizer prompt carries every OK DEBIT; OK
    # CREDITs are rule-assigned Income and never sent to the categorizer.
    debit_ids = [t["id"] for t in ingested if t["status"] == "OK" and t["direction"] == "DEBIT"]
    credit_ids = [t["id"] for t in ingested if t["status"] == "OK" and t["direction"] == "CREDIT"]
    assert len(debit_ids) == 6 and len(credit_ids) == 2
    categorizer_prompts = [p for p in prompts if phase3_categorized.ROWS_START in p]
    assert len(categorizer_prompts) == 1
    assert all(i in categorizer_prompts[0] for i in debit_ids)
    assert not any(i in categorizer_prompts[0] for i in credit_ids)
    # Q6: the plan phase's sample rows are non-refund DEBITs too.
    plan_prompt = next(p for p in prompts if "plan_steps" in p)
    credit_merchants = [t["merchant"] for t in ingested if t["status"] == "OK" and t["direction"] == "CREDIT"]
    assert credit_merchants == ["Oakwood HOA", "Riverside Apartments"]
    assert not any(m in plan_prompt for m in credit_merchants)
    assert "Shell Gas" in plan_prompt  # positive control: DEBIT sample rows are there

    categorized = json.loads((outputs / "categorized.json").read_text())["categorized"]
    assert sorted(c["id"] for c in categorized) == sorted(ok_ids)
    assert dq["kpi_join"]["rows_in_kpis"] == 8


PII_IN_FIXTURE = ["212-555-0147", "jane.doe@example.com", "4111 1111 1111 1111", "123456789012",
                  "(415) 555-0100", "3782 822463 10005", "billing+ops@gusto.com", "4111-1111-1111-1111",
                  "+1 415 555 0100"]
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}")
# Card or account shapes: 8+ contiguous digits, or 4-digit groups (dates like
# 2024-10-03 are allowed in prompts and don't match).
LONG_DIGITS_RE = re.compile(r"\d{8,}|\d{4}[ -]\d{4,6}[ -]\d{4,5}")


def test_no_pii_pattern_in_any_prompt(tmp_path, monkeypatch):
    """PII in DEBIT descriptions (card, account, phone, email), in a DEBIT
    merchant that becomes top merchant, and in a CREDIT merchant that becomes
    top client, so all four prompts (plan, categorize, summary, reflection)
    are exercised."""
    prompts = stub_model_calls(monkeypatch)
    biz = business_dir(tmp_path, os.path.join(FIXTURES, "pipeline", "pii.csv"))
    assert run.main(business_dir=biz) == 0

    outputs = tmp_path / "outputs"
    ingested = json.loads((outputs / "ingested.json").read_text())["transactions"]
    assert [t["status"] for t in ingested] == ["OK"] * 10
    ids = [t["id"] for t in ingested]
    assert len(prompts) == 4

    for prompt in prompts:
        for pii in PII_IN_FIXTURE:
            assert pii not in prompt
        text = prompt
        for i in ids:  # ids are hex and may contain digit runs; they are not PII
            text = text.replace(i, "<id>")
        assert not EMAIL_RE.search(text)
        assert not PHONE_RE.search(text)
        assert not LONG_DIGITS_RE.search(text), LONG_DIGITS_RE.search(text).group()

    # Positive controls: masked forms arrive where the PII was.
    categorizer = next(p for p in prompts if phase3_categorized.ROWS_START in p)
    summary = next(p for p in prompts if "financial reporting agent" in p)
    assert "****1111" in categorizer and "****9012" in categorizer and "[PHONE]" in categorizer
    assert "Zelle [EMAIL]" in summary  # top merchant
    assert "Wire from [PHONE]" in summary  # top client

    # Local files keep the original text.
    kept = (outputs / "ingested.json").read_text() + (outputs / "transactions.json").read_text()
    for pii in PII_IN_FIXTURE:
        assert pii in kept


def test_h2_failed_ingest_stops_before_any_llm_call(tmp_path, llm_calls, capsys):
    biz = business_dir(tmp_path, os.path.join(FIXTURES, "ingest", "threshold_fail.csv"))

    assert run.main(business_dir=biz) == 1

    assert llm_calls == []
    assert list((tmp_path / "outputs").iterdir()) == []
    out = capsys.readouterr().out
    assert "FAILED at Ingest" in out
    assert "20.0%" in out
