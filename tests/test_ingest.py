"""
Tests for afw/ingest.py and afw/models.py — validated CSV ingest.

Each fixture in tests/fixtures/ingest/ has one row per edge case. Tests look
rows up by source_row (the physical line number, header = line 1), so the
row-to-expectation mapping matches the table in the Phase 1 plan:
edge_cases.csv (A), positive_is_credit.csv (B), refunds.csv (C),
threshold_*.csv / empty.csv (D), cp1252.csv (E).
"""
import json
import os
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from afw import ingest
from afw.models import Direction, SignConvention, SourceConfig, Status, Transaction

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ingest")
SNAPSHOTS = os.path.join(os.path.dirname(__file__), "fixtures", "snapshots")

OK, REVIEW, REJECTED = Status.OK, Status.NEEDS_REVIEW, Status.REJECTED
DEBIT, CREDIT = Direction.DEBIT, Direction.CREDIT


def fixture(name):
    return os.path.join(FIXTURES, name)


def by_row(result):
    return {t.source_row: t for t in result.rows}


# ------------------------------------------------------ A. edge_cases.csv

@pytest.fixture(scope="module")
def edge():
    return ingest.ingest_rows(fixture("edge_cases.csv"), SourceConfig())


# (line, status, direction, amount, reason)
EDGE_CASES = [
    (2, OK, CREDIT, "6500.00", None),                   # A1  1dp, negative = CREDIT
    (3, OK, DEBIT, "420.50", None),                     # A2  1dp normalized
    (4, OK, DEBIT, "19.99", None),                      # A3  trailing zero allowed
    (5, REJECTED, None, None, "amount_precision"),      # A4
    (6, REJECTED, None, None, "amount_zero"),           # A5
    (7, REJECTED, None, None, "amount_zero"),           # A6  -0.00
    (8, REJECTED, None, None, "amount_blank"),          # A7
    (9, REJECTED, None, None, "amount_unparseable"),    # A8
    (10, REJECTED, None, None, "non_usd"),              # A9  currency column
    (11, REJECTED, None, None, "non_usd"),              # A10 currency symbol
    (12, REJECTED, None, None, "date_format"),          # A11 ISO in MM-DD-YYYY source
    (13, REJECTED, None, None, "date_format"),          # A12 slashes
    (14, REJECTED, None, None, "date_invalid"),         # A13 Feb 30
    (15, REVIEW, DEBIT, "55.00", "merchant_blank"),     # A14
    (16, OK, DEBIT, "80.00", None),                     # A15
    (18, OK, DEBIT, "6.25", None),                      # A16
    (20, OK, DEBIT, "4.50", None),                      # A17
    (21, OK, DEBIT, "4.50", None),                      # A18
    (22, OK, DEBIT, "12.00", None),                     # A21
]


@pytest.mark.parametrize("line,status,direction,amount,reason", EDGE_CASES)
def test_edge_case_row(edge, line, status, direction, amount, reason):
    t = by_row(edge)[line]
    assert t.status == status
    assert t.direction == direction
    assert t.amount == (Decimal(amount) if amount else None)
    assert t.reason == reason


def test_blank_rows_are_skipped_not_counted(edge):
    # A19 (",,,,") is line 17 and A20 (empty line) is line 19.
    lines = {t.source_row for t in edge.rows}
    assert 17 not in lines and 19 not in lines
    assert len(edge.rows) == 19
    assert edge.counts == {"rows": 19, "ok": 8, "needs_review": 1, "rejected": 10}


def test_amounts_are_normalized_to_two_places(edge):
    for t in edge.rows:
        if t.amount is not None:
            assert t.amount.as_tuple().exponent == -2


def test_merchant_whitespace_trimmed(edge):
    assert by_row(edge)[16].merchant == "Home Depot"


def test_bom_stripped_and_utf8_kept(edge):
    a1 = by_row(edge)[2]
    assert a1.date == date(2024, 10, 1)  # would be unparsed if the header were "﻿date"
    assert by_row(edge)[18].merchant == "Café Nero"
    assert edge.warnings == []


def test_row_metadata(edge):
    t = by_row(edge)[2]
    assert t.source_file == "edge_cases.csv"
    assert t.source_row == 2
    assert t.account_id == "default"
    assert isinstance(t.date, date)


def test_identical_rows_get_distinct_ids(edge):
    a17, a18 = by_row(edge)[20], by_row(edge)[21]
    assert a17.id != a18.id
    assert a17.id.rsplit("-", 1)[0] == a18.id.rsplit("-", 1)[0]  # same content hash
    assert (a17.id.endswith("-0"), a18.id.endswith("-1")) == (True, True)


def test_ids_unique_within_file(edge):
    ids = [t.id for t in edge.rows]
    assert len(ids) == len(set(ids))


def test_description_truncated_to_cap(edge):
    assert len(by_row(edge)[22].description) == 500
    assert edge.description_truncated == 1


def test_transfer_never_set_in_phase1(edge):
    assert not any(t.is_transfer for t in edge.rows)


# ---------------------------------------------- B. sign convention

def test_positive_is_credit_source():
    cfg = SourceConfig(sign_convention=SignConvention.POSITIVE_IS_CREDIT)
    rows = by_row(ingest.ingest_rows(fixture("positive_is_credit.csv"), cfg))
    assert (rows[2].status, rows[2].direction, rows[2].amount) == (OK, CREDIT, Decimal("100.00"))
    assert (rows[3].status, rows[3].direction, rows[3].amount) == (OK, DEBIT, Decimal("40.00"))


def test_unknown_sign_convention_rejected_by_config():
    with pytest.raises(ValidationError):
        SourceConfig(sign_convention="sometimes_credit")


def test_unsupported_date_format_rejected_by_config():
    with pytest.raises(ValidationError):
        SourceConfig(date_format="%b %d, %Y")


# ------------------------------------------------------------ C. refunds

@pytest.fixture(scope="module")
def refunds():
    return by_row(ingest.ingest_rows(fixture("refunds.csv"), SourceConfig()))


# (line, status, direction, amount, is_refund, refund_of line, reason)
REFUND_CASES = [
    (2, OK, DEBIT, "200.00", False, None, None),                           # C1
    (3, OK, CREDIT, "50.00", True, 2, None),                               # C2 case-insensitive merchant
    (4, OK, CREDIT, "150.00", True, 2, None),                              # C3 uses the rest of C1
    (5, REVIEW, CREDIT, "10.00", False, None, "refund_merchant_match_only"),   # C4 C1 used up
    (6, REVIEW, CREDIT, "300.00", False, None, "refund_merchant_match_only"),  # C5 bigger than any DEBIT
    (7, REVIEW, CREDIT, "40.00", False, None, "refund_merchant_match_only"),   # C6 DEBIT comes later
    (8, OK, DEBIT, "40.00", False, None, None),                            # C7
    (9, REVIEW, CREDIT, "25.00", False, None, "refund_keyword_only"),      # C8
    (10, OK, CREDIT, "6500.00", False, None, None),                        # C9 ordinary income
    (11, OK, CREDIT, "20.00", True, 12, None),                             # C10 same-day DEBIT counts
    (12, OK, DEBIT, "20.00", False, None, None),                           # C11
    (13, OK, CREDIT, "100.00", False, None, None),                         # C12 'return' is not a keyword
]


@pytest.mark.parametrize("line,status,direction,amount,is_refund,refund_of,reason", REFUND_CASES)
def test_refund_row(refunds, line, status, direction, amount, is_refund, refund_of, reason):
    t = refunds[line]
    assert t.status == status
    assert t.direction == direction
    assert t.amount == Decimal(amount)
    assert t.is_refund is is_refund
    assert t.refund_of == (refunds[refund_of].id if refund_of else None)
    assert t.reason == reason


def test_needs_review_totals():
    result = ingest.ingest_rows(fixture("refunds.csv"), SourceConfig())
    dq = ingest.data_quality(result)
    assert dq["needs_review_total"] == {"debit": "0.00", "credit": "375.00", "rows": 4}
    assert dq["reasons"] == {"refund_merchant_match_only": 3, "refund_keyword_only": 1}


# ------------------------------------------------------------ D. threshold

def test_exactly_ten_percent_rejected_passes():
    result = ingest.ingest_file(fixture("threshold_pass.csv"), SourceConfig())
    assert result.counts == {"rows": 10, "ok": 9, "needs_review": 0, "rejected": 1}


def test_over_ten_percent_rejected_fails(tmp_path):
    with pytest.raises(ingest.IngestFailed, match=r"20\.0%"):
        ingest.main(csv_path=fixture("threshold_fail.csv"), outputs_dir=str(tmp_path))
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_needs_review_does_not_count_toward_threshold():
    result = ingest.ingest_file(fixture("threshold_review.csv"), SourceConfig())
    assert result.counts == {"rows": 10, "ok": 5, "needs_review": 5, "rejected": 0}


def test_blank_rows_excluded_from_threshold_denominator():
    with pytest.raises(ingest.IngestFailed, match=r"11\.1%"):
        ingest.ingest_file(fixture("threshold_blanks.csv"), SourceConfig())


def test_header_only_file_fails():
    with pytest.raises(ingest.IngestFailed, match="no data rows"):
        ingest.ingest_file(fixture("empty.csv"), SourceConfig())


# ------------------------------------------------------------ E. encoding

def test_cp1252_fallback_with_warning():
    result = ingest.ingest_rows(fixture("cp1252.csv"), SourceConfig())
    assert result.rows[0].merchant == "Café Nero"
    assert result.warnings == ["encoding_fallback_cp1252"]
    assert ingest.data_quality(result)["warnings"] == ["encoding_fallback_cp1252"]


# ------------------------------------------------------ model invariants

def test_ok_row_requires_amount():
    with pytest.raises(ValidationError):
        Transaction(id="x-0", source_file="f.csv", source_row=2, account_id="default",
                    date=date(2024, 10, 1), direction=DEBIT, merchant="A", status=OK)


def test_rejected_row_requires_reason():
    with pytest.raises(ValidationError):
        Transaction(id="x-0", source_file="f.csv", source_row=2, account_id="default",
                    status=REJECTED)


# -------------------------------------------------------- main() / outputs

def test_main_writes_ingested_and_data_quality(tmp_path):
    ingest.main(csv_path=fixture("threshold_review.csv"), outputs_dir=str(tmp_path))

    ingested = json.loads((tmp_path / "ingested.json").read_text())
    assert ingested["source_file"] == "threshold_review.csv"
    first = ingested["transactions"][0]
    assert first["date"] == "2024-10-01"
    assert first["amount"] == "11.00"
    assert first["direction"] == "DEBIT"
    assert first["status"] == "OK"

    dq = json.loads((tmp_path / "data_quality.json").read_text())
    assert dq["counts"] == {"rows": 10, "ok": 5, "needs_review": 5, "rejected": 0}
    assert dq["reasons"] == {"merchant_blank": 5}
    assert dq["needs_review_total"] == {"debit": "90.00", "credit": "0.00", "rows": 5}
    assert dq["description_truncated"] == 0
    assert dq["warnings"] == []


@pytest.mark.parametrize("business,rows", [("landscaper", 42), ("law_firm", 35), ("restaurant", 47)])
def test_demo_businesses_ingest_clean(business, rows):
    path = os.path.join(SNAPSHOTS, business, "transactions.csv")
    result = ingest.ingest_file(path, ingest.config_for(path))
    assert result.counts == {"rows": rows, "ok": rows, "needs_review": 0, "rejected": 0}


def test_config_for_unknown_folder_is_default():
    assert ingest.config_for("businesses/_uploads/abc123/transactions.csv") == SourceConfig()
