"""
Tests for eval/make_gold_set.py and the committed eval/gold_set.csv — the
hand-labeled purchases the eval scores against. The generator selects rows
with production's own rule (OK, non-refund DEBITs) and never writes labels;
labels are added by hand.
"""
import csv
import os

import pytest

from afw import ingest
from afw.llm_input import model_rows
from afw.prompt_versions import PROMPTS
from eval import make_gold_set as gold

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMITTED = os.path.join(REPO, "eval", "gold_set.csv")
HEADER = ["id", "business", "merchant", "description", "amount", "label"]
EXPECTED_COUNTS = {"landscaper": 32, "law_firm": 30, "restaurant": 34, "consultant": 46}


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    path = tmp_path_factory.mktemp("gold") / "gold_set.csv"
    gold.main(path=str(path))
    return read(path)


def test_header(generated):
    assert generated[0] == HEADER


def test_every_purchase_from_all_four_businesses(generated):
    rows = generated[1:]
    assert len(rows) == 142
    counts = {b: sum(1 for r in rows if r[1] == b) for b in EXPECTED_COUNTS}
    assert counts == EXPECTED_COUNTS


@pytest.mark.parametrize("business", list(EXPECTED_COUNTS))
def test_rows_are_exactly_what_production_sends_to_the_model(generated, business):
    path = os.path.join(REPO, "businesses", business, "transactions.csv")
    ingested = [t.model_dump(mode="json") for t in ingest.ingest_file(path, ingest.config_for(path)).rows]
    ok = [t for t in ingested if t["status"] == "OK"]
    expected = [[t["id"], business, t["merchant"], t["description"], t["amount"], ""] for t in model_rows(ok)]
    assert [r for r in generated[1:] if r[1] == business] == expected


def test_excludes_credits_refunds_review_and_rejected_rows(generated):
    path = os.path.join(REPO, "businesses", "consultant", "transactions.csv")
    rows = ingest.ingest_file(path, ingest.config_for(path)).rows
    excluded = {t.id for t in rows if t.status.value != "OK" or t.direction.value == "CREDIT" or t.is_refund}
    assert len(excluded) == 8  # 5 client payments, 1 refund, 1 blank merchant, 1 REJECTED
    assert not excluded & {r[0] for r in generated[1:]}


def test_labels_are_left_empty(generated):
    assert all(r[5] == "" for r in generated[1:])


def test_refuses_to_overwrite_a_labeled_file(tmp_path, generated):
    path = tmp_path / "gold_set.csv"
    labeled = [generated[0], generated[1][:5] + ["Shopping"], *generated[2:]]
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(labeled)
    before = path.read_bytes()
    with pytest.raises(SystemExit, match="already has labels"):
        gold.main(path=str(path))
    assert path.read_bytes() == before


def test_regenerating_an_unlabeled_file_is_byte_identical(tmp_path):
    path = tmp_path / "gold_set.csv"
    gold.main(path=str(path))
    first = path.read_bytes()
    gold.main(path=str(path))
    assert path.read_bytes() == first


# ------------------------------------------------ the committed gold set


def test_committed_gold_set_matches_the_data(generated):
    """If a demo CSV changes, this fails: regenerate before labeling, or
    relabel the rows that changed (there's no merge mode)."""
    assert [r[:5] for r in read(COMMITTED)] == [r[:5] for r in generated]


def test_committed_labels_are_empty_or_valid_categories():
    valid = {c.value for c in PROMPTS["v2"].categories}  # the finest set any version scores against
    bad = [(r[0], r[5]) for r in read(COMMITTED)[1:] if r[5] and r[5] not in valid]
    assert not bad, f"labels must be one of {sorted(valid)}: {bad}"
