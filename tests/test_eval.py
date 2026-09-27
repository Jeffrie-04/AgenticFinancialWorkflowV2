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
IDENTITY = ("stub_provider", "stub-model-1", "https://stub.example/v1")


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


def flaky_provider(monkeypatch, *outcomes):
    """The real call_model (with its retries), its provider call replaced:
    each outcome in turn is raised or returned. Nothing sleeps."""
    import bedrock_client
    sent = []

    def provider(prompt):
        outcome = outcomes[len(sent)]
        sent.append(prompt)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    monkeypatch.setattr(bedrock_client, "MODEL_PROVIDER", "openai")
    monkeypatch.setattr(bedrock_client, "_call_openai", provider)
    monkeypatch.setattr(bedrock_client, "_sleep", lambda seconds: None)
    return bedrock_client.call_model, sent


def rate_limited():
    import httpx2
    import openai
    request = httpx2.Request("POST", "https://stub.example/v1")
    return openai.RateLimitError("rate limited", response=httpx2.Response(429, request=request), body=None)


def test_cache_stores_only_the_final_reply_after_a_retry(tmp_path, monkeypatch):
    call, sent = flaky_provider(monkeypatch, rate_limited(), "the reply")
    stats = run_eval.Counter(calls=0, cache_hits=0)
    cached = run_eval.cached(call, IDENTITY, str(tmp_path), stats)

    assert cached("the prompt") == "the reply"
    assert sent == ["the prompt", "the prompt"]  # retried inside call_model
    assert stats["calls"] == 1                   # one call as the cache sees it
    files = os.listdir(tmp_path)
    assert len(files) == 1
    assert (tmp_path / files[0]).read_text(encoding="utf-8") == "the reply"

    assert cached("the prompt") == "the reply"  # and it replays from the cache
    assert (stats["calls"], stats["cache_hits"], len(sent)) == (1, 1, 2)


def test_cache_writes_nothing_when_retries_run_out(tmp_path, monkeypatch):
    import openai
    call, sent = flaky_provider(monkeypatch, *[rate_limited() for _ in range(4)])
    stats = run_eval.Counter(calls=0, cache_hits=0)
    cached = run_eval.cached(call, IDENTITY, str(tmp_path / "cache"), stats)

    with pytest.raises(openai.RateLimitError):
        cached("the prompt")
    assert len(sent) == 4
    assert not (tmp_path / "cache").exists()


def test_cache_key_changes_with_the_rows_the_version_and_the_model(tmp_path, dirs):
    answers = {"a1": TRAVEL, "a2": "Shopping", "a3": "Utilities", "b1": TRAVEL, "b2": "Dining", "b3": "Other"}
    run(tmp_path, dirs, "v2", dict(answers))
    changed = [("a1", "alpha", "Delta Air Lines", "Flight to Boston", "412.60", TRAVEL), *GOLD[1:]]
    result, _ = run(tmp_path, dirs, "v2", dict(answers), changed)
    assert (result["calls"], result["cache_hits"]) == (1, 1)  # alpha changed, beta didn't

    key = run_eval.cache_key
    assert key(IDENTITY, "PROMPT") != key(("stub_provider", "stub-model-2", "https://stub.example/v1"), "PROMPT")
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
    assert (saved["provider"], saved["model"], saved["endpoint"]) == IDENTITY
    assert len(saved["gold_sha256"]) == 64


# ------------------------------------------------------------------ --offline

ANSWERS = {"a1": TRAVEL, "a2": "Shopping", "a3": "Utilities", "b1": TRAVEL, "b2": "Dining", "b3": "Other"}


@pytest.fixture
def no_live_model(monkeypatch):
    """Offline mode must never reach the real model or read the env identity."""
    import bedrock_client

    def refuse(*args, **kwargs):
        raise AssertionError("offline mode reached the live model or the env identity")
    monkeypatch.setattr(bedrock_client, "call_model", refuse)
    monkeypatch.setattr(run_eval, "model_identity", refuse)


def record_online(tmp_path, dirs, monkeypatch, version="v2", generated_at="2000-01-02"):
    """A normal (online) run with a stub model, then pin generated_at to a past date."""
    gold = write_gold(tmp_path / "gold.csv", GOLD)
    with monkeypatch.context() as m:
        m.setattr(run_eval, "model_identity", lambda: IDENTITY)
        run_eval.main(["--version", version, "--gold", gold], call=Model(dict(ANSWERS)), **dirs)
    path = os.path.join(dirs["results_dir"], f"{version}.json")
    with open(path) as f:
        saved = json.load(f)
    saved["generated_at"] = generated_at
    with open(path, "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2)
    return gold, path, saved


def test_offline_replays_from_the_cache_with_the_recorded_identity_and_date(tmp_path, dirs, monkeypatch,
                                                                            no_live_model):
    gold, path, recorded = record_online(tmp_path, dirs, monkeypatch)

    run_eval.main(["--version", "v2", "--gold", gold, "--offline"], **dirs)

    with open(path) as f:
        replayed = json.load(f)
    assert (replayed["provider"], replayed["model"], replayed["endpoint"]) == IDENTITY
    assert replayed["generated_at"] == "2000-01-02"  # when the replies were generated, not today
    assert (replayed["calls"], replayed["cache_hits"]) == (0, 2)
    assert {k: v for k, v in replayed.items() if k not in ("calls", "cache_hits")} == \
        {k: v for k, v in recorded.items() if k not in ("calls", "cache_hits")}


def test_offline_cache_miss_is_an_error_and_calls_nothing(tmp_path, dirs, monkeypatch, no_live_model):
    gold, path, recorded = record_online(tmp_path, dirs, monkeypatch)
    changed = [row if row[0] != "a2" else ("a2", "alpha", "Office Depot", "Toner", "80.00", "Shopping")
               for row in GOLD]
    gold = write_gold(tmp_path / "gold.csv", changed)  # alpha's prompt changes: a miss

    with pytest.raises(SystemExit) as raised:
        run_eval.main(["--version", "v2", "--gold", gold, "--offline"], **dirs)

    message = str(raised.value)
    assert message.startswith("offline: cache miss for v2 (")
    assert "run without --offline" in message
    with open(path) as f:
        assert json.load(f) == recorded  # results left untouched
    assert len(os.listdir(os.path.join(dirs["cache_dir"], "v2"))) == 2  # nothing added


def test_offline_ignores_the_environment_identity(tmp_path, dirs, monkeypatch, no_live_model):
    import bedrock_client
    gold, path, _ = record_online(tmp_path, dirs, monkeypatch)
    monkeypatch.setattr(bedrock_client, "MODEL_PROVIDER", "claude")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://elsewhere.example/v1")

    run_eval.main(["--version", "v2", "--gold", gold, "--offline"], **dirs)

    with open(path) as f:
        assert json.load(f)["provider"] == "stub_provider"


def test_offline_without_recorded_results_is_an_error(tmp_path, dirs, no_live_model):
    gold = write_gold(tmp_path / "gold.csv", GOLD)
    with pytest.raises(SystemExit) as raised:
        run_eval.main(["--version", "v2", "--gold", gold, "--offline"], **dirs)
    assert "no recorded results" in str(raised.value)
    assert not os.path.exists(dirs["cache_dir"])


def test_offline_requires_a_version(tmp_path, dirs):
    with pytest.raises(SystemExit):
        run_eval.main(["--offline"], **dirs)


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


# ------------------------------------------------ review fixes (Codex)


def test_same_id_in_two_businesses_is_scored_per_business(tmp_path, dirs):
    # Content-hash ids can repeat across businesses (same account, same row
    # content); predictions must not overwrite each other.
    gold = [("x1", "alpha", "Staples", "Paper", "10.00", "Shopping"),
            ("x1", "beta", "Chipotle", "Lunch", "12.00", "Dining")]

    def model(prompt):
        category = "Shopping" if '"merchant": "Staples"' in prompt else "Dining"
        return json.dumps({"categorized": [{"id": "x1", "category": category}]})
    result = run_eval.evaluate(write_gold(tmp_path / "gold.csv", gold), "v2", call=model,
                               identity=IDENTITY, cache_dir=dirs["cache_dir"])
    assert result["accuracy"] == 1.0  # not 50%


def test_duplicate_id_within_a_business_is_rejected(tmp_path, dirs):
    gold = [("x1", "alpha", "Staples", "Paper", "10.00", "Shopping"),
            ("x1", "alpha", "Staples", "Paper", "10.00", "Shopping")]
    with pytest.raises(SystemExit, match="duplicate ids within a business: alpha/x1"):
        run(tmp_path, dirs, "v2", {}, gold)


def test_f1_is_zero_when_travel_is_all_errors(tmp_path, dirs):
    gold = [("t1", "alpha", "Delta Air Lines", "Flight", "412.60", TRAVEL),
            ("o1", "alpha", "FedEx", "Shipping", "27.45", "Other")]
    result, _ = run(tmp_path, dirs, "v2", {"t1": "Other", "o1": TRAVEL}, gold)
    assert result["travel"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                                "predicted": 1, "gold": 1, "true_positive": 0}


def test_f1_is_none_only_without_any_travel_data(tmp_path, dirs):
    gold = [("s1", "alpha", "Staples", "Paper", "64.35", "Shopping")]
    result, _ = run(tmp_path, dirs, "v2", {"s1": "Shopping"}, gold)
    assert result["travel"] == {"precision": None, "recall": None, "f1": None,
                                "predicted": 0, "gold": 0, "true_positive": 0}


EIGHT = [
    ("r1", "alpha", "Delta Air Lines", "Flight", "412.60", TRAVEL),  # -> Travel   (TP)
    ("r2", "alpha", "Uber", "Ride", "31.15", TRAVEL),                # -> Other    (FN)
    ("r3", "alpha", "Hertz", "Rental", "146.80", TRAVEL),            # -> review   (FN)
    ("r4", "alpha", "FedEx", "Shipping", "27.45", "Other"),          # -> Travel   (FP)
    ("r5", "alpha", "Staples", "Paper", "64.35", "Shopping"),        # -> Shopping
    ("r6", "alpha", "Chipotle", "Lunch", "13.85", "Dining"),         # -> Dining
    ("r7", "alpha", "Verizon", "Phone", "85.00", "Utilities"),       # -> Utilities
    ("r8", "alpha", "Jiffy Lube", "Oil change", "69.99", "Other"),   # -> review
]
EIGHT_ANSWERS = {"r1": TRAVEL, "r2": "Other", "r3": [None, None], "r4": TRAVEL,
                 "r5": "Shopping", "r6": "Dining", "r7": "Utilities", "r8": [None, None]}


def test_codex_eight_row_gold_set_every_number(tmp_path, dirs):
    result, _ = run(tmp_path, dirs, "v2", EIGHT_ANSWERS, EIGHT)
    assert (result["labeled"], result["correct"], result["answered"]) == (8, 4, 6)
    assert result["accuracy"] == 0.5
    assert result["answered_accuracy"] == pytest.approx(4 / 6)
    assert run_eval._pct(result["answered_accuracy"]) == "66.7%"
    assert result["review_counts"] == {"llm_missing": 2}
    assert {c: v["accuracy"] for c, v in result["per_category"].items()} == pytest.approx(
        {"Utilities": 1.0, "Shopping": 1.0, "Dining": 1.0, TRAVEL: 1 / 3, "Other": 0.0})
    assert result["travel"]["precision"] == 0.5
    assert result["travel"]["recall"] == pytest.approx(1 / 3)
    assert result["travel"]["f1"] == pytest.approx(0.4)
    assert [run_eval._pct(result["travel"][k]) for k in ("precision", "recall", "f1")] == ["50.0%", "33.3%", "40.0%"]


def test_initial_and_repair_calls_replay_fully_from_cache(tmp_path, dirs):
    answers = {"a1": [None, TRAVEL], "a2": "Shopping", "a3": "Utilities",
               "b1": TRAVEL, "b2": "Dining", "b3": "Other"}
    first, _ = run(tmp_path, dirs, "v2", answers)
    assert (first["calls"], first["cache_hits"]) == (3, 0)  # alpha, alpha's repair, beta

    def no_model(prompt):
        raise AssertionError("cache miss on replay")
    second = run_eval.evaluate(write_gold(tmp_path / "gold.csv", GOLD), "v2", call=no_model,
                               identity=IDENTITY, cache_dir=dirs["cache_dir"])
    assert (second["calls"], second["cache_hits"]) == (0, 3)
    assert second["rows"] == first["rows"] and second["accuracy"] == first["accuracy"] == 1.0


def test_endpoint_is_part_of_the_cache_identity(tmp_path, dirs, monkeypatch):
    import bedrock_client
    monkeypatch.setattr(bedrock_client, "MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://mantle.example/v1")
    first = run_eval.model_identity()
    assert first == ("openai", bedrock_client.OPENAI_MODEL_ID, "https://mantle.example/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://other.example/v1")
    second = run_eval.model_identity()
    assert run_eval.cache_key(first, "PROMPT") != run_eval.cache_key(second, "PROMPT")

    answers = {"a1": TRAVEL, "a2": "Shopping", "a3": "Utilities", "b1": TRAVEL, "b2": "Dining", "b3": "Other"}
    gold = write_gold(tmp_path / "gold.csv", GOLD)
    for identity, expected_calls in ((first, 2), (first, 0), (second, 2)):
        result = run_eval.evaluate(gold, "v2", call=Model(dict(answers)), identity=identity,
                                   cache_dir=dirs["cache_dir"])
        assert result["calls"] == expected_calls


def test_identity_never_includes_api_keys(monkeypatch):
    import bedrock_client
    monkeypatch.setattr(bedrock_client, "MODEL_PROVIDER", "anthropic_direct")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-value")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-other-secret")
    assert not any("secret" in part for part in run_eval.model_identity())


# ------------------------------------------------ before/after label correction


def scored_pair(tmp_path, dirs, gold_rows, name):
    gold = write_gold(tmp_path / f"{name}.csv", gold_rows)
    answers = {"a1": TRAVEL, "a2": "Shopping", "a3": "Utilities", "b1": "Other", "b2": "Dining", "b3": "Other"}
    return [run_eval.evaluate(gold, v, call=Model(dict(answers)), identity=IDENTITY, cache_dir=dirs["cache_dir"])
            for v in ("v1", "v2")]


def test_compare_shows_before_and_after_and_the_audit_note(tmp_path, dirs):
    corrected = scored_pair(tmp_path, dirs, GOLD, "after")
    original = scored_pair(tmp_path, dirs, [*GOLD[:5], ("b3", "beta", "FedEx", "Shipping", "27.45", "Shopping")],
                           "before")
    text = run_eval.results_markdown(*corrected, before=original, note="Labels were corrected by rule.")
    after_part, before_part = text.split("## With the original labels", 1)
    assert "| Overall accuracy | 83.3% | 83.3% |" in after_part           # corrected labels
    assert "| Overall accuracy | 66.7% | 66.7% |" in before_part          # b3 was labeled Shopping
    assert original[1]["gold_sha256"] in before_part and corrected[1]["gold_sha256"] in after_part
    assert "## Label audit\n\nLabels were corrected by rule." in text


def test_compare_cli_picks_up_original_labels_and_audit_note(tmp_path, dirs):
    after = scored_pair(tmp_path, dirs, GOLD, "after")
    before = scored_pair(tmp_path, dirs, GOLD, "before")
    os.makedirs(os.path.join(dirs["results_dir"], "original_labels"))
    for result, sub in ((after, ""), (before, "original_labels")):
        for r in result:
            with open(os.path.join(dirs["results_dir"], sub, f"{r['version']}.json"), "w") as f:
                json.dump(r, f)
    with open(os.path.join(dirs["results_dir"], "label_audit.md"), "w") as f:
        f.write("Audit note.\n")
    md = tmp_path / "RESULTS.md"
    run_eval.main(["--compare"], results_md=str(md), **dirs)
    text = md.read_text()
    assert "## With the original labels" in text and "## Label audit\n\nAudit note." in text


def test_compare_without_before_or_note_is_unchanged(tmp_path, dirs):
    v1, v2 = scored_pair(tmp_path, dirs, GOLD, "only")
    text = run_eval.results_markdown(v1, v2)
    assert "original labels" not in text and "Label audit" not in text
