"""
Tests for afw/narrative.py and the summary/reflection phases: model text
must be grounded in the KPIs; otherwise it is regenerated once, and then
replaced by a deterministic fallback. The outcome is recorded in
data_quality.json. call_model is always stubbed.
"""
import json
import os
import re
import shutil
import subprocess

import pytest

import phase3_reflection
import phase3_summary
from afw.guards.grounding import (
    FALLBACK_REFLECTION_MARKER,
    FALLBACK_SUMMARY_MARKER,
    check_grounding,
    fallback_reflection,
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
    text, outcome, _ = grounded_text("PROMPT", KPIS, fallback_summary, call)
    assert text == GOOD_SUMMARY.strip()
    assert outcome == {"status": "grounded", "attempts": 1, "unsupported": []}
    assert len(prompts) == 1


def test_ungrounded_reply_is_regenerated_once_with_a_fixed_message():
    call, prompts = scripted(BAD_SUMMARY, GOOD_SUMMARY)
    text, outcome, _ = grounded_text("PROMPT", KPIS, fallback_summary, call)
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
    text, outcome, _ = grounded_text("PROMPT", KPIS, fallback_summary, call)
    assert text == fallback_summary(KPIS)
    assert text.startswith(FALLBACK_SUMMARY_MARKER)
    assert check_grounding(text, KPIS).ok
    assert outcome == {"status": "fallback", "attempts": 2, "unsupported": ["33%"]}
    assert len(prompts) == 2


@pytest.mark.parametrize("empty", ["", "   \n", None])
def test_empty_or_non_text_reply_counts_as_a_failure(empty):
    call, _ = scripted(empty, empty)
    text, outcome, _ = grounded_text("PROMPT", KPIS, fallback_summary, call)
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
    _, outcome, _ = grounded_text("PROMPT", kpis, fallback_summary, call)
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


# ------------------------------------------ display KPIs in prompts; attempts


@pytest.mark.parametrize("module,replies", [(phase3_summary, [GOOD_SUMMARY]),
                                            (phase3_reflection, ["Your business is healthy."])])
def test_prompts_show_display_formatted_kpis(outputs, monkeypatch, module, replies):
    prompts, _ = run_phase(module, outputs, monkeypatch, *replies)
    assert '"total_income": "$58,860.00"' in prompts[0]
    assert '"pct_of_spend": "51.1%"' in prompts[0]
    assert '"period_days": 28' in prompts[0]
    assert "58860.0" not in prompts[0]  # no raw JSON floats for the model to copy


def test_prompts_still_mask_names(outputs, monkeypatch):
    data = json.loads((outputs / "kpis.json").read_text())
    data["kpis"]["top_merchants"][0] = "Zelle jane@x.com"
    (outputs / "kpis.json").write_text(json.dumps(data))
    prompts, _ = run_phase(phase3_summary, outputs, monkeypatch, GOOD_SUMMARY)
    assert "jane@x.com" not in prompts[0] and "Zelle [EMAIL]" in prompts[0]


def attempts(outputs):
    return json.loads((outputs / "narrative_attempts.json").read_text())


def test_rejected_attempts_are_saved_for_debugging(outputs, monkeypatch):
    run_phase(phase3_summary, outputs, monkeypatch, BAD_SUMMARY, GOOD_SUMMARY)
    assert attempts(outputs)["summary"] == [{"attempt": 1, "text": BAD_SUMMARY.strip(), "unsupported": ["8.0%"]}]

    bad = "Income was about 59,000 and spend about 48,000."
    run_phase(phase3_reflection, outputs, monkeypatch, bad, "Still about 2 things.")
    saved = attempts(outputs)
    assert saved["reflection"] == [{"attempt": 1, "text": bad, "unsupported": ["59,000", "48,000"]},
                                   {"attempt": 2, "text": "Still about 2 things.", "unsupported": ["2"]}]
    assert saved["summary"][0]["attempt"] == 1  # the other narrative's attempts are kept


def test_grounded_run_clears_stale_attempts(outputs, monkeypatch):
    run_phase(phase3_summary, outputs, monkeypatch, BAD_SUMMARY, GOOD_SUMMARY)
    run_phase(phase3_summary, outputs, monkeypatch, GOOD_SUMMARY)
    assert attempts(outputs)["summary"] == []


def test_attempts_file_is_never_read_into_a_prompt():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    readers = []
    for name in os.listdir(repo):
        if name.endswith(".py"):
            with open(os.path.join(repo, name)) as f:
                if "narrative_attempts" in f.read():
                    readers.append(name)
    for root, _, files in os.walk(os.path.join(repo, "afw")):
        for name in files:
            if name.endswith(".py"):
                with open(os.path.join(root, name)) as f:
                    if "narrative_attempts" in f.read():
                        readers.append(name)
    assert readers == ["narrative.py"]  # only the writer knows the file


def test_attempts_file_is_git_ignored():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in ("businesses/landscaper/outputs/narrative_attempts.json", "outputs/narrative_attempts.json"):
        result = subprocess.run(["git", "check-ignore", "-q", path], cwd=repo, check=False)
        assert result.returncode == 0, path


def test_reflection_prompt_steps_are_unnumbered_bullets(outputs, monkeypatch):
    prompts, _ = run_phase(phase3_reflection, outputs, monkeypatch, "Your business is healthy.")
    prompt = prompts[0]
    assert not re.search(r"(?m)^\s*\d+[.)]\s", prompt)  # nothing numbered for the model to echo
    for step in ("- Review all provided KPIs and understand the business's position.",
                 "- Identify which numbers are concerning and which are strong.",
                 "- Consider what likely drove those numbers (which categories, clients, costs).",
                 "- Open with a one-line overall verdict on the period (e.g. healthy surplus,",
                 "- State the most important observations, each tied to a specific KPI.",
                 "- Give specific, actionable recommendations tied to those observations —"):
        assert step in prompt
    assert "Keep the response under 150 words." in prompt  # word limit kept


# ---------------------------------------------- review hardening (Codex)


def test_zero_width_only_reply_regenerates_then_falls_back():
    call, prompts = scripted("\u200b", "\u200b\ufeff ")
    text, outcome, rejected = grounded_text("PROMPT", KPIS, fallback_summary, call)
    assert len(prompts) == 2
    assert outcome["status"] == "fallback"
    assert [r["text"] for r in rejected] == ["", ""]
    assert text == fallback_summary(KPIS)


def test_an_ungrounded_fallback_is_flagged():
    call, _ = scripted("Made up 12%.", "Made up 13%.")
    text, outcome, _ = grounded_text("PROMPT", KPIS, lambda kpis: "Fallback with 99%.", call)
    assert text == "Fallback with 99%."
    assert outcome["status"] == "fallback_ungrounded"


def test_real_fallbacks_are_never_flagged_ungrounded():
    from tests.test_grounding import ALL_KPIS
    for name, kpis in ALL_KPIS.items():
        for fallback in (fallback_summary, fallback_reflection):
            call, _ = scripted("Made up 12%.", "Made up 13%.")
            _, outcome, _ = grounded_text("PROMPT", kpis, fallback, call)
            assert outcome["status"] == "fallback", (name, fallback.__name__)
