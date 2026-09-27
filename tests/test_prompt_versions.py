"""
Tests for afw/prompt_versions.py — categorizer prompts are versioned text
files in prompts/. Each version fixes its own category list, so adding an
enum value never changes an existing prompt. v1 is the production prompt;
tests/fixtures/prompts/categorizer_prompt.txt pins its rendered bytes.
"""
import json

import pytest

import phase3_categorized as cat_mod
from afw.guards.output_validation import check_reply
from afw.models import Category
from afw.prompt_versions import PRODUCTION_VERSION, PROMPTS, prompt_version


def template_text(version):
    with open(PROMPTS[version].path, encoding="utf-8") as f:
        return f.read()


def allowed_line(text):
    line = next(l for l in text.splitlines() if l.startswith("Allowed categories (use these exact strings): "))
    return line.split(": ", 1)[1].split(", ")


def test_production_is_v1():
    assert PRODUCTION_VERSION == "v1"


@pytest.mark.parametrize("version", list(PROMPTS))
def test_allowed_line_matches_the_version_categories(version):
    names = allowed_line(template_text(version))
    assert names == [c.value for c in PROMPTS[version].categories]
    assert all(n in {c.value for c in Category} for n in names)
    assert Category.INCOME.value not in names  # only DEBITs are sent; Income is by rule


@pytest.mark.parametrize("version", list(PROMPTS))
def test_guide_lines_name_only_the_version_categories(version):
    guide = [l.split(":", 1)[0][2:] for l in template_text(version).splitlines()
             if l.startswith("- ") and ":" in l]
    assert guide == [c.value for c in PROMPTS[version].categories]


@pytest.mark.parametrize("version", list(PROMPTS))
def test_template_has_one_block_between_one_delimiter_pair(version):
    lines = template_text(version).splitlines()
    assert lines.count(cat_mod.ROWS_START) == 1 and lines.count(cat_mod.ROWS_END) == 1
    assert lines[lines.index(cat_mod.ROWS_START) + 1] == "$block"
    assert template_text(version).count("$") == 1


def test_v1_categories_are_fixed_not_derived_from_the_enum():
    assert [c.value for c in PROMPTS["v1"].categories] == ["Utilities", "Shopping", "Dining", "Other"]


def test_unknown_version_is_a_clear_error():
    with pytest.raises(ValueError, match="unknown prompt version 'v9'"):
        prompt_version("v9")


def test_build_prompt_defaults_to_production():
    rows = [{"id": "a", "merchant": "Shop", "description": "x", "direction": "DEBIT"}]
    assert cat_mod.build_prompt(rows) == cat_mod.build_prompt(rows, version=PRODUCTION_VERSION)


def test_production_main_sends_the_v1_prompt(tmp_path, monkeypatch):
    rows = [{"id": "d1", "merchant": "Shop", "description": "Supplies", "direction": "DEBIT", "amount": "1.00",
             "date": "2024-10-01", "status": "OK", "reason": None, "is_refund": False, "refund_of": None}]
    (tmp_path / "ingested.json").write_text(json.dumps({"source_file": "t.csv", "transactions": rows}))
    prompts = []
    monkeypatch.setattr(cat_mod, "call_model",
                        lambda p: prompts.append(p) or '{"categorized": [{"id": "d1", "category": "Shopping"}]}')
    cat_mod.main(outputs_dir=str(tmp_path))
    assert prompts == [cat_mod.build_prompt(rows, version="v1")]


def test_categorize_takes_a_version_and_an_injected_model_call():
    rows = [{"id": "d1", "merchant": "Shop", "description": "x", "direction": "DEBIT"}]
    prompts = []

    def call(prompt):
        prompts.append(prompt)
        return '{"categorized": [{"id": "d1", "category": "Dining"}]}'
    accepted, failures, counts, error = cat_mod.categorize(rows, version="v1", call=call)
    assert (accepted, failures, counts["calls"], error) == ({"d1": "Dining"}, {}, 1, None)
    assert prompts == [cat_mod.build_prompt(rows, version="v1")]


# -------------------------------------------- check_reply: version's allowed set

SENT = {"a": {"id": "a", "direction": "DEBIT"}}


def test_enum_category_not_offered_by_the_version_is_invalid():
    result = check_reply({"categorized": [{"id": "a", "category": "Dining"}]}, SENT, allowed={"Shopping"})
    assert result.failures == {"a": "llm_invalid_category"}


def test_direction_is_still_checked_before_the_allowed_set():
    result = check_reply({"categorized": [{"id": "a", "category": "Income"}]}, SENT, allowed={"Shopping"})
    assert result.failures == {"a": "category_direction_mismatch"}


def test_repair_text_lists_the_version_categories():
    assert cat_mod.problem_text("llm_invalid_category", "v1") == \
        "category is not one of: Utilities, Shopping, Dining, Other"
