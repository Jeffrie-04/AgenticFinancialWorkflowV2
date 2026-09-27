"""
Tests for eval/run_eval.py — the categorizer eval harness. Every test uses a
hand-made gold set in a temp dir and a stubbed model; nothing here reads
eval/gold_set.csv or calls a real model.
"""
import csv
import json
import os

import pytest

import phase3_categorized as cat_mod
from eval import run_eval

TRAVEL = "Travel/Transportation"
IDENTITY = ("stub_provider", "stub-model-1")


def write_gold(path, rows):
    """rows: (id, business, merchant, description, amount, label)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "business", "merchant", "description", "amount", "label"])
        w.writerows(rows)
    return str(path)


def block_ids(prompt):
    block = prompt.split(f"\n{cat_mod.ROWS_START}\n", 1)[1].split(f"\n{cat_mod.ROWS_END}", 1)[0]
    return [json.loads(line)["id"] for line in block.splitlines()]


class Model:
    """A stub model: answers each id from `answers` (id -> category, or a
    list of per-call categories), and records every prompt it receives."""

    def __init__(self, answers):
        self.answers, self.prompts = answers, []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        items = []
        for i in block_ids(prompt):
            answer = self.answers.get(i)
            if isinstance(answer, list):
                answer = answer.pop(0)
            if answer is not None:
                items.append({"id": i, "category": answer})
        return json.dumps({"categorized": items})


@pytest.fixture
def dirs(tmp_path):
    return {"cache_dir": str(tmp_path / "cache"), "results_dir": str(tmp_path / "results")}


GOLD = [
    ("a1", "alpha", "Delta Air Lines", "Flight", "412.60", TRAVEL),
    ("a2", "alpha", "Staples", "Paper", "64.35", "Shopping"),
    ("a3", "alpha", "Verizon", "Phone", "85.00", "Utilities"),
    ("b1", "beta", "Uber", "Ride", "31.15", TRAVEL),
    ("b2", "beta", "Chipotle", "Lunch", "13.85", "Dining"),
    ("b3", "beta", "FedEx", "Shipping", "27.45", "Other"),
]


def run(tmp_path, dirs, version, answers, gold=GOLD):
    model = Model(answers)
    result = run_eval.evaluate(write_gold(tmp_path / "gold.csv", gold), version, call=model,
                               identity=IDENTITY, cache_dir=dirs["cache_dir"])
    return result, model


# ------------------------------------------------------------------ scoring


def test_scoring_math_v2(tmp_path, dirs):
    answers = {"a1": TRAVEL, "a2": "Shopping", "a3": "Other", "b1": "Other", "b2": "Dining", "b3": "Other"}
    result, model = run(tmp_path, dirs, "v2", answers)
    assert len(model.prompts) == 2  # one request per business, as in production
    assert result["labeled"] == 6 and result["unlabeled"] == 0
    assert result["correct"] == 4
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["answered_accuracy"] == pytest.approx(4 / 6)
    assert result["per_category"][TRAVEL] == {"support": 2, "correct": 1, "accuracy": 0.5}
    assert result["per_category"]["Utilities"] == {"support": 1, "correct": 0, "accuracy": 0.0}
    assert result["confusion"][TRAVEL] == {TRAVEL: 1, "Other": 1}
    assert result["confusion"]["Utilities"] == {"Other": 1}
    # Travel/Transportation: 1 predicted, 1 correct; 2 in gold.
    assert result["travel"] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(2 / 3),
                                "predicted": 1, "gold": 2, "true_positive": 1}


def test_v1_scores_travel_gold_labels_as_other(tmp_path, dirs):
    answers = {"a1": "Other", "a2": "Shopping", "a3": "Utilities", "b1": "Other", "b2": "Dining", "b3": "Other"}
    result, _ = run(tmp_path, dirs, "v1", answers)
    assert result["correct"] == 6 and result["accuracy"] == 1.0
    assert TRAVEL not in result["per_category"]
    assert result["per_category"]["Other"] == {"support": 3, "correct": 3, "accuracy": 1.0}
    assert result["travel"] is None  # v1 has no Travel/Transportation to measure


def test_v1_reply_of_travel_is_a_review_row_not_a_match(tmp_path, dirs):
    answers = {"a1": [TRAVEL, TRAVEL], "a2": "Shopping", "a3": "Utilities",
               "b1": "Other", "b2": "Dining", "b3": "Other"}
    result, _ = run(tmp_path, dirs, "v1", answers)
    row = next(r for r in result["rows"] if r["id"] == "a1")
    assert row["predicted"] == "REVIEW:llm_invalid_category"
    assert result["review_counts"] == {"llm_invalid_category": 1}


def test_review_rows_count_as_incorrect_with_answered_accuracy_alongside(tmp_path, dirs):
    answers = {"a1": TRAVEL, "a2": "Shopping", "a3": "Utilities", "b1": TRAVEL, "b2": "Dining",
               "b3": [None, None]}  # never answered, even after the retry
    result, _ = run(tmp_path, dirs, "v2", answers)
    assert result["accuracy"] == pytest.approx(5 / 6)
    assert result["answered"] == 5 and result["answered_accuracy"] == 1.0
    assert result["review_counts"] == {"llm_missing": 1}
    assert result["confusion"]["Other"] == {"REVIEW": 1}


def test_retry_path_is_scored(tmp_path, dirs):
    answers = {"a1": [None, TRAVEL], "a2": "Shopping", "a3": "Utilities",
               "b1": TRAVEL, "b2": "Dining", "b3": "Other"}
    result, model = run(tmp_path, dirs, "v2", answers)
    assert len(model.prompts) == 3  # alpha, alpha's repair for a1, beta
    assert block_ids(model.prompts[1]) == ["a1"]
    assert result["accuracy"] == 1.0
    assert result["calls"] == 3


# -------------------------------------------------------------- gold set rules


def test_unlabeled_rows_are_skipped_and_counted(tmp_path, dirs):
    gold = [*GOLD[:2], ("a3", "alpha", "Verizon", "Phone", "85.00", "")]
    result, model = run(tmp_path, dirs, "v2", {"a1": TRAVEL, "a2": "Shopping"}, gold)
    assert (result["labeled"], result["unlabeled"]) == (2, 1)
    assert all("a3" not in p for p in model.prompts)  # never sent


def test_invalid_labels_are_an_error_listing_the_ids(tmp_path, dirs):
    gold = [*GOLD[:2], ("a3", "alpha", "Verizon", "Phone", "85.00", "Travel"),
            ("a4", "alpha", "Hertz", "Car", "146.80", "Income")]
    with pytest.raises(SystemExit, match=r"a3: 'Travel'.*a4: 'Income'"):
        run(tmp_path, dirs, "v2", {}, gold)


def test_nothing_labeled_is_an_error(tmp_path, dirs):
    gold = [(i, b, m, d, a, "") for i, b, m, d, a, _ in GOLD]
    with pytest.raises(SystemExit, match="no labeled rows"):
        run(tmp_path, dirs, "v2", {}, gold)


def test_uses_the_real_categorizer_path_with_masking(tmp_path, dirs):
    gold = [("a1", "alpha", "Hilton Garden Inn", "card 4242 4242 4242 4242 on file", "178.50", TRAVEL)]
    _, model = run(tmp_path, dirs, "v2", {"a1": TRAVEL}, gold)
    assert "4242 4242 4242 4242" not in model.prompts[0]
    assert "card ****4242 on file" in model.prompts[0]
    assert "Dining, Travel/Transportation, Other" in model.prompts[0]  # the v2 prompt


# -------------------------------------------------------------------- cache


def test_cache_miss_calls_and_stores_then_rerun_is_free(tmp_path, dirs):
    answers = {"a1": TRAVEL, "a2": "Shopping", "a3": "Utilities", "b1": TRAVEL, "b2": "Dining", "b3": "Other"}
    first, model = run(tmp_path, dirs, "v2", dict(answers))
    assert (first["calls"], first["cache_hits"], len(model.prompts)) == (2, 0, 2)
    assert len(os.listdir(os.path.join(dirs["cache_dir"], "v2"))) == 2

    second, model = run(tmp_path, dirs, "v2", dict(answers))
    assert (second["calls"], second["cache_hits"], len(model.prompts)) == (0, 2, 0)
    assert {k: v for k, v in second.items() if k not in ("calls", "cache_hits")} == \
        {k: v for k, v in first.items() if k not in ("calls", "cache_hits")}


def test_cache_key_changes_with_the_rows_the_version_and_the_model(tmp_path, dirs):
    answers = {"a1": TRAVEL, "a2": "Shopping", "a3": "Utilities", "b1": TRAVEL, "b2": "Dining", "b3": "Other"}
    run(tmp_path, dirs, "v2", dict(answers))
    changed = [("a1", "alpha", "Delta Air Lines", "Flight to Boston", "412.60", TRAVEL), *GOLD[1:]]
    result, _ = run(tmp_path, dirs, "v2", dict(answers), changed)
    assert (result["calls"], result["cache_hits"]) == (1, 1)  # alpha changed, beta didn't

    key = run_eval.cache_key
    assert key(IDENTITY, "PROMPT") != key(("stub_provider", "stub-model-2"), "PROMPT")
    assert key(IDENTITY, "PROMPT") != key(IDENTITY, "PROMPT ")


# ------------------------------------------------------------ files and CLI


def test_main_writes_results_json(tmp_path, dirs, monkeypatch):
    answers = {"a1": TRAVEL, "a2": "Shopping", "a3": "Utilities", "b1": TRAVEL, "b2": "Dining", "b3": "Other"}
    gold = write_gold(tmp_path / "gold.csv", GOLD)
    monkeypatch.setattr(run_eval, "model_identity", lambda: IDENTITY)
    run_eval.main(["--version", "v2", "--gold", gold], call=Model(answers), **dirs)
    with open(os.path.join(dirs["results_dir"], "v2.json")) as f:
        saved = json.load(f)
    assert saved["version"] == "v2" and saved["accuracy"] == 1.0
    assert (saved["provider"], saved["model"]) == IDENTITY
    assert len(saved["gold_sha256"]) == 64


def test_compare_writes_results_md(tmp_path, dirs):
    gold = write_gold(tmp_path / "gold.csv", GOLD)
    v1 = run_eval.evaluate(gold, "v1", call=Model({"a1": "Other", "a2": "Shopping", "a3": "Utilities",
                                                   "b1": "Other", "b2": "Dining", "b3": "Shopping"}),
                           identity=IDENTITY, cache_dir=dirs["cache_dir"])
    v2 = run_eval.evaluate(gold, "v2", call=Model({"a1": TRAVEL, "a2": "Shopping", "a3": "Utilities",
                                                   "b1": "Other", "b2": "Dining", "b3": "Other"}),
                           identity=IDENTITY, cache_dir=dirs["cache_dir"])
    text = run_eval.results_markdown(v1, v2)
    assert "| Overall accuracy | 83.3% | 83.3% |" in text
    assert "| Travel/Transportation precision | — | 100.0% |" in text
    assert "| Travel/Transportation recall | — | 50.0% |" in text
    assert "single run, pinned by cache" in text
    assert v2["gold_sha256"] in text and "stub-model-1" in text


def test_compare_refuses_mismatched_gold_sets(tmp_path, dirs):
    v1 = {"version": "v1", "gold_sha256": "a" * 64}
    v2 = {"version": "v2", "gold_sha256": "b" * 64}
    with pytest.raises(SystemExit, match="different gold sets"):
        run_eval.results_markdown(v1, v2)
