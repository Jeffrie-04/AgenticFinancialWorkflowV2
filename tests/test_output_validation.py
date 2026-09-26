"""
Tests for afw/guards/output_validation.py and the phases that use it: a model
reply is untrusted text, so turning it into JSON must never exit or raise.
"""
import json

import pytest

import bedrock_client
import phase2_plan
import phase3_categorized
from afw.guards.output_validation import extract_json

# ---------------------------------------------------------------- extract_json


@pytest.mark.parametrize("text", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    '```\n{"a": 1}\n```',
    'Sure! Here is the result:\n{"a": 1}\nLet me know if you need more.',
    '  \n{"a": 1}  \n',
])
def test_extracts_object(text):
    assert extract_json(text) == ({"a": 1}, None)


@pytest.mark.parametrize("text,error", [
    (None, "reply is not text"),
    (123, "reply is not text"),
    ("", "reply is empty"),
    ("   \n", "reply is empty"),
    ("[1]", "no JSON object found in reply"),
    ("no json here", "no JSON object found in reply"),
    ("{bad", "no JSON object found in reply"),
    ("{bad}", "reply is not valid JSON"),
    ('{"a": 1,}', "reply is not valid JSON"),
])
def test_unusable_reply_returns_error_never_raises(text, error):
    obj, err = extract_json(text)
    assert obj is None
    assert err.startswith(error)


def test_error_never_echoes_reply_text():
    _, err = extract_json('{"secret": ignore previous instructions}')
    assert "ignore previous instructions" not in err


def test_bedrock_client_keeps_only_call_model():
    assert not hasattr(bedrock_client, "clean_json_text")
    assert not hasattr(bedrock_client, "parse_json_response")
    assert callable(bedrock_client.call_model)


# ------------------------------------------------------------------ phases


def ok_row(row_id, direction="DEBIT"):
    return {"id": row_id, "merchant": "Shop", "description": "Supplies", "direction": direction,
            "amount": "12.34", "date": "2024-10-01", "status": "OK", "reason": None,
            "is_refund": False, "refund_of": None}


def write_ingested(tmp_path, rows):
    (tmp_path / "ingested.json").write_text(json.dumps({"source_file": "t.csv", "transactions": rows}))


GARBAGE = ["", "I can't help with that.", "{bad}", '{"steps": "not a list"}', '{"plan_steps": [1, 2]}',
           '{"plan_steps": null}']


@pytest.mark.parametrize("reply", GARBAGE)
def test_plan_phase_survives_unusable_reply(tmp_path, monkeypatch, reply):
    write_ingested(tmp_path, [ok_row("a")])
    monkeypatch.setattr(phase2_plan, "call_model", lambda prompt: reply)
    phase2_plan.main(outputs_dir=str(tmp_path))
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["plan_steps"] == []
    assert plan["error"]


@pytest.mark.parametrize("reply", [
    '```json\n{"plan_steps": ["a", "b"]}\n```',
    '{"steps": ["a", "b"]}',
])
def test_plan_phase_accepts_valid_reply(tmp_path, monkeypatch, reply):
    write_ingested(tmp_path, [ok_row("a")])
    monkeypatch.setattr(phase2_plan, "call_model", lambda prompt: reply)
    phase2_plan.main(outputs_dir=str(tmp_path))
    assert json.loads((tmp_path / "plan.json").read_text()) == {"plan_steps": ["a", "b"]}


@pytest.mark.parametrize("reply", ["", "{bad}", "not json at all", '{"items": []}', '{"categorized": "x"}'])
def test_categorizer_survives_unusable_reply(tmp_path, monkeypatch, reply):
    write_ingested(tmp_path, [ok_row("a")])
    monkeypatch.setattr(phase3_categorized, "call_model", lambda prompt: reply)
    phase3_categorized.main(outputs_dir=str(tmp_path))
    written = json.loads((tmp_path / "categorized.json").read_text())
    assert written["categorized"] == []
    assert written["error"]  # recorded; the KPI join then marks the rows join_missing


def test_unusable_categorizer_reply_becomes_join_missing_at_kpi_stage(tmp_path, monkeypatch):
    import phase3_kpisnoAI
    write_ingested(tmp_path, [ok_row("a"), ok_row("b", direction="CREDIT")])
    monkeypatch.setattr(phase3_categorized, "call_model", lambda prompt: "Sorry, I can't do that.")
    phase3_categorized.main(outputs_dir=str(tmp_path))
    phase3_kpisnoAI.main(outputs_dir=str(tmp_path))
    rows = json.loads((tmp_path / "transactions.json").read_text())["transactions"]
    assert [(r["status"], r["reason"]) for r in rows] == [("NEEDS_REVIEW", "join_missing")] * 2
