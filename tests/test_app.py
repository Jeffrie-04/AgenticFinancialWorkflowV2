"""
Tests for app.py — the dashboard renders from cached files without crashing,
whatever state the outputs are in, and never calls a model on page load.

Each test builds businesses/law_firm/outputs/ (the default selection) in a
temp dir and runs the real app.py there with streamlit's AppTest.
"""
import ast
import json
import os
import shutil
import subprocess

import pytest
from streamlit.testing.v1 import AppTest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(REPO, "app.py")
OLD_FORMAT_CATEGORIZED = os.path.join(REPO, "tests", "fixtures", "categorized_sample.json")

INGESTED = {"source_file": "transactions.csv", "transactions": [
    {"id": "ok1-0", "date": "2024-10-02", "merchant": "Shell Gas", "direction": "DEBIT",
     "amount": "420.50", "status": "OK", "reason": None},
    {"id": "ok2-0", "date": "2024-10-01", "merchant": "Oakwood HOA", "direction": "CREDIT",
     "amount": "6500.00", "status": "OK", "reason": None},
    {"id": "nr-0", "date": "2024-10-03", "merchant": "", "direction": "DEBIT",
     "amount": "55.00", "status": "NEEDS_REVIEW", "reason": "merchant_blank"},
    {"id": "rj-0", "date": None, "merchant": "Mystery Vendor", "direction": None,
     "amount": None, "status": "REJECTED", "reason": "amount_unparseable"},
]}
CATEGORIZED = {"categorized": [{"id": "ok1-0", "category": "Other"}, {"id": "ok2-0", "category": "Income"}]}
DATA_QUALITY = {"counts": {"rows": 4, "ok": 2, "needs_review": 1, "rejected": 1}}


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    out = tmp_path / "businesses" / "law_firm" / "outputs"
    out.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    def no_pipeline(*args, **kwargs):
        raise AssertionError("page load shelled out to the pipeline")
    monkeypatch.setattr(subprocess, "run", no_pipeline)
    return out


def write(out, name, data):
    (out / name).write_text(json.dumps(data) if not isinstance(data, str) else data)


def render():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_new_format_shows_every_row_with_status_and_category(outputs):
    write(outputs, "ingested.json", INGESTED)
    write(outputs, "categorized.json", CATEGORIZED)
    write(outputs, "data_quality.json", DATA_QUALITY)
    at = render()

    table = at.dataframe[0].value
    assert list(table["status"]) == ["NEEDS_REVIEW", "OK", "OK", "REJECTED"]  # newest first, undated last
    rows = table.set_index("merchant")
    assert rows.loc["Shell Gas", "category"] == "Other"
    assert rows.loc["Shell Gas", "amount"] == 420.50
    assert rows.loc["Mystery Vendor", "reason"] == "amount_unparseable"
    assert "Data quality: 2 OK · 1 needs review · 1 rejected" in [c.value for c in at.caption]
    assert not [w for w in at.warning if "categorized" in w.value]


def test_old_format_categorized_warns_and_still_shows_rows(outputs):
    write(outputs, "ingested.json", INGESTED)
    shutil.copy(OLD_FORMAT_CATEGORIZED, outputs / "categorized.json")
    at = render()

    assert any("old format" in w.value for w in at.warning)
    table = at.dataframe[0].value
    assert len(table) == 4
    assert table["category"].isna().all()


def test_missing_categorized_warns_and_still_shows_rows(outputs):
    write(outputs, "ingested.json", INGESTED)
    at = render()
    assert any("categorized.json not found" in w.value for w in at.warning)
    assert len(at.dataframe[0].value) == 4


def test_missing_ingested_warns(outputs):
    write(outputs, "categorized.json", CATEGORIZED)
    at = render()
    assert any("ingested.json not found" in w.value for w in at.warning)


@pytest.mark.parametrize("name,content", [
    ("ingested.json", "{not json"),
    ("ingested.json", {"transactions": [None]}),
    ("categorized.json", {"categorized": "nope"}),
])
def test_malformed_files_warn_instead_of_crashing(outputs, name, content):
    write(outputs, "ingested.json", INGESTED)
    write(outputs, "categorized.json", CATEGORIZED)
    write(outputs, name, content)
    at = render()
    assert any(name in w.value for w in at.warning)


def test_empty_outputs_folder_renders(outputs):
    render()


def test_app_never_imports_pipeline_or_model_code():
    with open(APP) as f:
        tree = ast.parse(f.read())
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert not imported & {"bedrock_client", "run", "afw", "openai", "anthropic", "boto3"}
    assert not any(name.startswith("phase") for name in imported)
