"""
Tests for eval/label.py — the interactive labeling tool. Every test works on
a temp file (built by the generator into a temp dir, or written by hand);
none reads or writes eval/gold_set.csv.
"""
import csv
import io
import os

import pytest

from eval import label as label_tool
from eval import make_gold_set

MENU = "1 Utilities  2 Shopping  3 Dining  4 Travel/Transportation  5 Other  s skip  q quit"


def keys(*pressed, on_press=None):
    """A scripted keyboard. on_press(n) runs before the n-th key is returned."""
    queue = list(pressed)
    count = [0]

    def read_key():
        count[0] += 1
        if on_press:
            on_press(count[0])
        if not queue:
            raise EOFError
        return queue.pop(0)
    return read_key


def label_session(path, *pressed, on_press=None):
    lines = []
    label_tool.run(str(path), read_key=keys(*pressed, on_press=on_press), out=lines.append)
    return "\n".join(lines)


def parse(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


@pytest.fixture
def gold(tmp_path):
    """A realistic unlabeled gold set (quoted descriptions included), generated
    from the demo CSVs into a temp dir: the real eval/gold_set.csv is never read."""
    path = tmp_path / "gold_set.csv"
    make_gold_set.main(path=str(path))
    return path


def test_keys_map_to_the_five_categories_in_menu_order():
    assert label_tool.CHOICES == {"1": "Utilities", "2": "Shopping", "3": "Dining",
                                  "4": "Travel/Transportation", "5": "Other"}


def test_writes_only_the_label_column_and_keeps_every_other_byte(gold):
    before = gold.read_bytes().decode("utf-8").split("\n")
    label_session(gold, "1", "4", "s", "5", "q")
    after = gold.read_bytes().decode("utf-8").split("\n")
    assert len(after) == len(before)
    changed = [(b, a) for b, a in zip(before, after) if b != a]
    assert [a[len(b):] for b, a in changed] == ["Utilities", "Travel/Transportation", "Other"]
    assert all(b.endswith(",") and a.startswith(b) for b, a in changed)  # label appended, nothing else
    rows = parse(gold)
    assert [r[5] for r in rows[1:5]] == ["Utilities", "Travel/Transportation", "", "Other"]


def test_saves_after_each_answer(gold):
    seen = {}

    def check(n):
        if n == 2:  # before the second key is read, the first answer is on disk
            seen["label"] = parse(gold)[1][5]
    label_session(gold, "2", "q", on_press=check)
    assert seen["label"] == "Shopping"


def test_resumes_at_the_first_unlabeled_row(gold):
    rows = parse(gold)
    label_session(gold, "1", "2", "q")
    output = label_session(gold, "q")
    shown = output.split("merchant:", 1)[1].splitlines()[0].strip()
    assert shown == rows[3][2]  # rows 1 and 2 are labeled, so row 3 is next
    assert "2 of 142 labeled" in output


def test_skipped_rows_stay_unlabeled_and_come_back_next_time(gold):
    rows = parse(gold)
    label_session(gold, "s", "3", "q")
    assert parse(gold)[1][5] == "" and parse(gold)[2][5] == "Dining"
    output = label_session(gold, "q")
    assert output.split("merchant:", 1)[1].splitlines()[0].strip() == rows[1][2]


def test_shows_business_merchant_description_amount_and_never_suggests(gold):
    rows = parse(gold)
    output = label_session(gold, "q")
    first = rows[1]
    for field, value in zip(("business", "merchant", "description", "amount"), first[1:5]):
        assert f"{field}: {value}" in output
    assert MENU in output
    body = output.replace(MENU, "")
    for category in label_tool.CHOICES.values():
        assert category not in body  # no category is shown except in the fixed menu
    assert parse(gold) == rows  # nothing pre-filled


def test_invalid_key_reprompts_and_writes_nothing(gold):
    before = gold.read_bytes()
    output = label_session(gold, "7", "x", "", "q")
    assert output.count("Press 1-5, s or q.") == 3
    assert gold.read_bytes() == before


def test_quit_or_end_of_input_without_answers_leaves_the_file_untouched(gold):
    before = gold.read_bytes()
    label_session(gold, "q")
    label_session(gold)  # EOF on the first key
    assert gold.read_bytes() == before


def test_interrupt_keeps_answers_already_saved(gold):
    def interrupt(n):
        if n == 2:
            raise KeyboardInterrupt
    label_session(gold, "5", "1", on_press=interrupt)
    assert parse(gold)[1][5] == "Other" and parse(gold)[2][5] == ""


def test_all_labeled_says_so(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("id,business,merchant,description,amount,label\na,x,Shop,Paper,1.00,Shopping\n")
    assert "All 1 rows are labeled." in label_session(path)


def test_crlf_quoted_and_multiline_fields_are_preserved_byte_for_byte(tmp_path):
    path = tmp_path / "g.csv"
    raw = ('id,business,merchant,description,amount,label\r\n'
           'a,x,"Hilton, Garden",' '"Rockford hotel, card on file",178.50,\r\n'
           'b,x,Uber,"two\nlines",31.15,""\r\n'
           'c,x,Staples,"say ""hi""",64.35,\r\n')
    path.write_bytes(raw.encode("utf-8"))
    label_session(path, "4", "4", "2")
    expected = raw.replace('178.50,\r\n', '178.50,Travel/Transportation\r\n') \
                  .replace('31.15,""\r\n', '31.15,Travel/Transportation\r\n') \
                  .replace('64.35,\r\n', '64.35,Shopping\r\n')
    assert path.read_bytes() == expected.encode("utf-8")


def test_refuses_a_file_whose_last_column_is_not_label(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("id,business,merchant,label,description,amount\na,x,Shop,,Paper,1.00\n")
    with pytest.raises(SystemExit, match="last column must be 'label'"):
        label_session(path, "1")


def test_default_path_is_the_real_gold_set_but_tests_never_use_it():
    assert label_tool.DEFAULT_PATH == os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "gold_set.csv")


def test_line_reader_fallback_reads_one_key_per_line(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("4\n q \n"))
    assert label_tool.read_line_key() == "4"
    assert label_tool.read_line_key() == "q"
    with pytest.raises(EOFError):
        label_tool.read_line_key()
