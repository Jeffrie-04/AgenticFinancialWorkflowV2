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

import phase3_kpisnoAI

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(REPO, "app.py")
OLD_FORMAT_CATEGORIZED = os.path.join(REPO, "tests", "fixtures", "categorized_sample.json")


def row(row_id, merchant, amount, direction, status="OK", reason=None, category=None, day="2024-10-02"):
    usable = status != "REJECTED"
    return {"id": row_id, "date": day if usable else None, "merchant": merchant,
            "direction": direction if usable else None, "amount": amount if usable else None,
            "description": "", "is_refund": False, "refund_of": None,
            "status": status, "reason": reason, "category": category}


TRANSACTIONS = {"transactions": [
    row("ok1-0", "Shell Gas", "420.50", "DEBIT", category="Other"),
    row("ok2-0", "Oakwood HOA", "6500.00", "CREDIT", category="Income", day="2024-10-01"),
    row("amb-0", "Costco", "30.00", "DEBIT", status="NEEDS_REVIEW", reason="join_ambiguous"),
    row("nr-0", "", "55.00", "DEBIT", status="NEEDS_REVIEW", reason="merchant_blank", day="2024-10-03"),
    row("rj-0", "Mystery Vendor", None, None, status="REJECTED", reason="amount_unparseable"),
]}


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


def transactions_table(at):
    tables = [d.value for d in at.dataframe if "status" in d.value.columns]
    assert len(tables) == 1
    return tables[0]


def table_rows(at):
    return transactions_table(at).set_index("merchant")


def test_table_shows_every_row_with_final_status_and_category(outputs):
    write(outputs, "transactions.json", TRANSACTIONS)
    at = render()

    table = transactions_table(at)
    assert list(table["status"]) == ["NEEDS_REVIEW", "OK", "NEEDS_REVIEW", "OK", "REJECTED"]  # newest first
    rows = table.set_index("merchant")
    assert rows.loc["Shell Gas", "category"] == "Other"
    assert rows.loc["Shell Gas", "amount"] == 420.50
    assert rows.loc["Costco", "reason"] == "join_ambiguous"
    assert rows.loc["Mystery Vendor", "reason"] == "amount_unparseable"
    # counts come from the same rows as the table
    assert "Data quality: 2 OK · 2 needs review · 1 rejected" in [c.value for c in at.caption]
    assert not [w for w in at.warning if "transactions.json" in w.value]


@pytest.mark.parametrize("reply", [["Shopping", "Dining"], ["Dining", "Shopping"]])
def test_kpis_and_dashboard_agree_on_codex_conflicting_categories(outputs, reply):
    ingested = [
        {"id": "rent-0", "source_file": "t.csv", "source_row": 2, "account_id": "default",
         "date": "2024-10-01", "amount": "1000.00", "direction": "DEBIT", "merchant": "Landlord",
         "description": "", "currency": "USD", "is_transfer": False, "is_refund": False,
         "refund_of": None, "status": "OK", "reason": None},
        {"id": "a-0", "source_file": "t.csv", "source_row": 3, "account_id": "default",
         "date": "2024-10-02", "amount": "10.00", "direction": "DEBIT", "merchant": "Shop",
         "description": "", "currency": "USD", "is_transfer": False, "is_refund": False,
         "refund_of": None, "status": "OK", "reason": None},
    ]
    write(outputs, "ingested.json", {"source_file": "t.csv", "transactions": ingested})
    write(outputs, "categorized.json", {"categorized": [{"id": "rent-0", "category": "Utilities"}] +
                                        [{"id": "a-0", "category": c} for c in reply]})
    kpis = phase3_kpisnoAI.main(outputs_dir=str(outputs))
    assert kpis["total_spend"] == 1000.0  # KPIs leave the conflicting row out...

    rows = table_rows(render())
    shop = rows.loc["Shop"]  # ...and the dashboard shows it flagged, not categorized
    assert (shop["status"], shop["reason"]) == ("NEEDS_REVIEW", "join_ambiguous")
    assert shop["category"] is None or shop["category"] != shop["category"]  # None / NaN
    assert rows.loc["Landlord", "category"] == "Utilities"


def test_missing_transactions_warns_and_never_falls_back_to_categorized(outputs):
    # Only the pre-KPI files exist (old-format categorized.json, as in the
    # committed demo outputs): warn, don't build a table from them.
    write(outputs, "ingested.json", {"transactions": [row("ok1-0", "Shell Gas", "420.50", "DEBIT")]})
    shutil.copy(OLD_FORMAT_CATEGORIZED, outputs / "categorized.json")
    at = render()
    assert any("transactions.json not found" in w.value for w in at.warning)
    assert not at.dataframe


@pytest.mark.parametrize("content", [
    "{not json",
    {"transactions": [None]},
    {"transactions": "nope"},
    {"transactions": [{"id": "x", "merchant": "Old row without status"}]},
])
def test_malformed_or_old_transactions_file_warns_instead_of_crashing(outputs, content):
    write(outputs, "transactions.json", content)
    at = render()
    assert any("transactions.json" in w.value for w in at.warning)
    assert not at.dataframe


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


def test_app_never_reads_categorized_json_directly():
    with open(APP) as f:
        assert "categorized.json" not in f.read()


def test_consultant_is_in_the_business_list_and_renders_without_outputs(outputs):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert "Consultant" in at.sidebar.selectbox[0].options
    at.sidebar.selectbox[0].select("Consultant").run()
    assert not at.exception
    assert any("not found yet" in w.value for w in at.warning)  # no outputs until a live run
