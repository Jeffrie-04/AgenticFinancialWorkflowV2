"""
tests/test_eval_replay.py — the committed eval replays offline, from the
committed cache, to the committed numbers. Reads eval/gold_set.csv,
eval/cache/ and eval/results/; writes only to a temp dir; never calls a model.

If this fails, eval/RESULTS.md (or results/*.json) no longer matches what
the code and the cached replies produce: re-run the eval and commit.
"""
import os
import shutil

import pytest

from eval import run_eval

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "eval", "results")
RESULTS_MD = os.path.join(REPO, "eval", "RESULTS.md")


@pytest.fixture
def replayed(tmp_path, monkeypatch):
    import bedrock_client

    def refuse(*args, **kwargs):
        raise AssertionError("the replay reached the live model or used the env identity")
    monkeypatch.setattr(bedrock_client, "call_model", refuse)
    monkeypatch.setattr(run_eval, "model_identity", refuse)

    results_dir = tmp_path / "results"
    shutil.copytree(RESULTS, results_dir)
    results_md = tmp_path / "RESULTS.md"
    dirs = {"cache_dir": run_eval.DEFAULT_CACHE, "results_dir": str(results_dir), "results_md": str(results_md)}
    cache_before = sorted(os.listdir(os.path.join(run_eval.DEFAULT_CACHE, "v1"))), \
        sorted(os.listdir(os.path.join(run_eval.DEFAULT_CACHE, "v2")))
    for version in ("v1", "v2"):
        run_eval.main(["--version", version, "--offline"], **dirs)
    run_eval.main(["--compare"], **dirs)
    cache_after = sorted(os.listdir(os.path.join(run_eval.DEFAULT_CACHE, "v1"))), \
        sorted(os.listdir(os.path.join(run_eval.DEFAULT_CACHE, "v2")))
    assert cache_after == cache_before  # a replay adds nothing to the committed cache
    return results_dir, results_md


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def test_replay_reproduces_the_committed_results_md(replayed):
    _, results_md = replayed
    assert read_bytes(results_md) == read_bytes(RESULTS_MD)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_replay_reproduces_the_committed_results_json(replayed, version):
    results_dir, _ = replayed
    assert read_bytes(results_dir / f"{version}.json") == read_bytes(os.path.join(RESULTS, f"{version}.json"))
