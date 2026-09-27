"""
Tests for afw/narrative.py and the summary/reflection phases: model text
must be grounded in the KPIs; otherwise it is regenerated once, and then
replaced by a deterministic fallback. The outcome is recorded in
data_quality.json. call_model is always stubbed.
"""
import json
import os
import shutil

import pytest

import phase3_reflection
import phase3_summary
from afw.guards.grounding import (
    FALLBACK_REFLECTION_MARKER,
    FALLBACK_SUMMARY_MARKER,
    check_grounding,
    fallback_summary,
)
from afw.narrative import COPY_EXACTLY, grounded_text

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
KPIS_FILE = os.path.join(FIXTURES, "snapshots", "landscaper", "kpis.json")
with open(KPIS_FILE) as f:
    KPIS = json.load(f)["kpis"]
with open(os.path.join(FIXTURES, "grounding", "landscaper_summary.txt"), encoding="utf-8") as f:
    BAD_SUMMARY = f.read()  # contains "The remaining 8.0%", a derived number
GOOD_SUMMARY = BAD_SUMMARY.replace("The remaining 8.0% was split between Other and Dining.",
                                   "Other took 7.0% and Dining 0.8%.")


def scripted(*replies):
    """A fake call_model that returns replies in order and records prompts."""
    prompts, queue = [], list(replies)

    def call(prompt):
        prompts.append(prompt)
        return queue.pop(0)
    return call, prompts


# ------------------------------------------------------------- grounded_text


def test_grounded_first_reply_makes_one_call():
    call, prompts = scripted(GOOD_SUMMARY)
    text, outcome = grounded_text("PROMPT", KPIS, fallback_summary, call)
    assert text == GOOD_SUMMARY.strip()
    assert outcome == {"status": "grounded", "attempts": 1, "unsupported": []}
    assert len(prompts) == 1


def test_ungrounded_reply_is_regenerated_once_with_a_fixed_message():
    call, prompts = scripted(BAD_SUMMARY, GOOD_SUMMARY)
    text, outcome = grounded_text("PROMPT", KPIS, fallback_summary, call)
    assert text == GOOD_SUMMARY.strip()
    assert outcome == {"status": "regenerated", "attempts": 2, "unsupported": ["8.0%"]}
    assert len(prompts) == 2
    retry = prompts[1]
    assert retry.startswith("PROMPT")
    assert ("Your previous text used numbers that are not in the KPIs: 8.0%. Rewrite it using only numbers "
            "copied exactly from the KPIs; do not round, add or compute.") in retry
    # Nothing the model wrote is echoed back, only our parsed token.
    assert "The remaining" not in retry and "Greenfield Estate Project" not in retry


def test_ungrounded_twice_gives_the_fallback():
    still_bad = "Utilities were about 51% and the rest made up 33%."  # 51% is 51.1 at 0 dp; 33% is no KPI
    call, prompts = scripted(BAD_SUMMARY, still_bad)
    text, outcome = grounded_text("PROMPT", KPIS, fallback_summary, call)
    assert text == fallback_summary(KPIS)
    assert text.startswith(FALLBACK_SUMMARY_MARKER)
    assert check_grounding(text, KPIS).ok
    assert outcome == {"status": "fallback", "attempts": 2, "unsupported": ["33%"]}
    assert len(prompts) == 2


@pytest.mark.parametrize("empty", ["", "   \n", None])
def test_empty_or_non_text_reply_counts_as_a_failure(empty):
    call, _ = scripted(empty, empty)
    text, outcome = grounded_text("PROMPT", KPIS, fallback_summary, call)
    assert outcome["status"] == "fallback"
    assert text == fallback_summary(KPIS)


def test_regenerate_message_masks_tokens_that_look_like_pii():
    call, prompts = scripted("Card 4111111111111111 was charged.", GOOD_SUMMARY)
    grounded_text("PROMPT", KPIS, fallback_summary, call)
    assert "4111111111111111" not in prompts[1]
    assert "****1111" in prompts[1]


def test_masked_names_the_model_saw_are_not_numbers():
    # The prompt shows the top merchant masked; the model repeats it that way.
    kpis = {**KPIS, "top_merchants": ["Zelle jane@x.com 5555 - 5555 - 5555 - 4444", "Toro Dealer", "Shell Gas"]}
    call, prompts = scripted("The largest merchant was Zelle [EMAIL] ****4444.")
    _, outcome = grounded_text("PROMPT", kpis, fallback_summary, call)
    assert outcome["status"] == "grounded"
    assert len(prompts) == 1


# ----------------------------------------------------------------- the phases


@pytest.fixture
def outputs(tmp_path):
    shutil.copy(KPIS_FILE, tmp_path / "kpis.json")
    (tmp_path / "data_quality.json").write_text(json.dumps({"counts": {"rows": 42}}))
    return tmp_path


def run_phase(module, outputs, monkeypatch, *replies):
    call, prompts = scripted(*replies)
    monkeypatch.setattr(module, "call_model", call)
    module.main(outputs_dir=str(outputs))
    return prompts, json.loads((outputs / "data_quality.json").read_text())


def test_summary_phase_regenerates_and_records(outputs, monkeypatch):
    prompts, dq = run_phase(phase3_summary, outputs, monkeypatch, BAD_SUMMARY, GOOD_SUMMARY)
    assert (outputs / "summary.txt").read_text() == GOOD_SUMMARY.strip()
    assert dq["grounding"]["summary"] == {"status": "regenerated", "attempts": 2, "unsupported": ["8.0%"]}
    assert dq["counts"] == {"rows": 42}  # other sections are kept
    assert COPY_EXACTLY in prompts[0]


def test_reflection_phase_falls_back_and_records(outputs, monkeypatch):
    bad = "Income was about 59,000 and spend about 48,000."
    prompts, dq = run_phase(phase3_reflection, outputs, monkeypatch, bad, bad)
    text = (outputs / "reflection.txt").read_text()
    assert text.startswith(FALLBACK_REFLECTION_MARKER)
    assert check_grounding(text, KPIS).ok
    assert dq["grounding"]["reflection"] == {"status": "fallback", "attempts": 2,
                                             "unsupported": ["59,000", "48,000"]}
    assert COPY_EXACTLY in prompts[0]
    assert "2-3" not in prompts[0]  # the prompt no longer invites a non-KPI number


def test_both_outcomes_recorded_side_by_side(outputs, monkeypatch):
    run_phase(phase3_summary, outputs, monkeypatch, GOOD_SUMMARY)
    _, dq = run_phase(phase3_reflection, outputs, monkeypatch, "Your business is healthy.")
    assert dq["grounding"] == {
        "summary": {"status": "grounded", "attempts": 1, "unsupported": []},
        "reflection": {"status": "grounded", "attempts": 1, "unsupported": []},
    }


def test_phase_creates_data_quality_if_missing(tmp_path, monkeypatch):
    shutil.copy(KPIS_FILE, tmp_path / "kpis.json")
    call, _ = scripted(GOOD_SUMMARY)
    monkeypatch.setattr(phase3_summary, "call_model", call)
    phase3_summary.main(outputs_dir=str(tmp_path))
    dq = json.loads((tmp_path / "data_quality.json").read_text())
    assert dq["grounding"]["summary"]["status"] == "grounded"
