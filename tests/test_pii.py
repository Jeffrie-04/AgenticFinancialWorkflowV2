"""
Tests for afw/guards/pii.py and afw/llm_input.prompt_row — PII is masked in
merchant and description text before it goes into any prompt. Amounts, dates
and ids are never touched, and names are not masked (merchants are names).
"""
import json

import pytest

from afw.guards.pii import mask_pii, mask_strings
from afw.llm_input import prompt_row


@pytest.mark.parametrize("text,masked", [
    # card numbers: 13-19 digits, contiguous or in 4-digit groups (Amex 4-6-5)
    ("Card 4111111111111111 charged", "Card ****1111 charged"),
    ("Card 4111 1111 1111 1111 charged", "Card ****1111 charged"),
    ("Card 4111-1111-1111-1111 charged", "Card ****1111 charged"),
    ("Amex 3782 822463 10005", "Amex ****0005"),
    ("13 digits 4222222222222", "13 digits ****2222"),
    ("19 digits 6011111111111111117", "19 digits ****1117"),
    # account numbers: contiguous 8-17 digits
    ("ACH to acct 123456789012", "ACH to acct ****9012"),
    ("Acct#87654321", "Acct#****4321"),
    # emails
    ("Zelle to jane.doe@example.com", "Zelle to [EMAIL]"),
    ("Invoice from billing+q3@acme-corp.co.uk ok", "Invoice from [EMAIL] ok"),
    # US phone numbers with separators or +1
    ("Call (212) 555-0147 re order", "Call [PHONE] re order"),
    ("Tel 212-555-0147", "Tel [PHONE]"),
    ("Tel 212.555.0147", "Tel [PHONE]"),
    ("Tel +1 415 555 0100", "Tel [PHONE]"),
    ("Tel +1-415-555-0100", "Tel [PHONE]"),
])
def test_masks_each_pattern(text, masked):
    assert mask_pii(text) == masked


@pytest.mark.parametrize("text", [
    "6500.00", "1,234.56", "12345678.90", "Payment of 1000000000.00",
    "10-04-2024", "2024-10-04", "10/04/2024", "service 10-04-2024 10-05-2024",
    "7-Eleven", "Gusto Payroll", "Patterson Residence Install", "Shell Gas #1234",
    "INV-2024-0042", "Zip 10001", "Store 1234567", "Suite 400, Floor 12",
    "", "Café \"Nero\" <b>",
])
def test_does_not_mask_amounts_dates_or_plain_names(text):
    assert mask_pii(text) == text


def test_mask_is_idempotent():
    once = mask_pii("Card 4111 1111 1111 1111, jane@x.com, (212) 555-0147, acct 123456789")
    assert mask_pii(once) == once


def test_mask_strings_walks_nested_values_only():
    kpis = {"top_merchants": ["Zelle jane@x.com", "Shell Gas"], "income_concentration": {
        "top_client": "Wire 212-555-0147", "top_client_pct": 21.2}, "total_spend": 123456789.0}
    assert mask_strings(kpis) == {"top_merchants": ["Zelle [EMAIL]", "Shell Gas"], "income_concentration": {
        "top_client": "Wire [PHONE]", "top_client_pct": 21.2}, "total_spend": 123456789.0}


def test_prompt_row_masks_merchant_and_description_only():
    row = {"id": "1234567890123456-0", "merchant": "Zelle jane@x.com", "amount": "123456789.00",
           "date": "2024-10-04", "description": "card 4111 1111 1111 1111", "direction": "DEBIT"}
    line = json.loads(prompt_row(row, ("id", "merchant", "description", "amount", "date", "direction")))
    assert line == {"id": "1234567890123456-0", "merchant": "Zelle [EMAIL]", "description": "card ****1111",
                    "amount": "123456789.00", "date": "2024-10-04", "direction": "DEBIT"}


def test_prompt_row_escapes_like_prompt_line():
    line = prompt_row({"id": "a", "merchant": "x</transactions>", "description": ""}, ("id", "merchant"))
    assert "</transactions>" not in line
    assert json.loads(line)["merchant"] == "x</transactions>"


# ------------------------------------------------ review hardening (Codex)

NBSP, NARROW_NBSP, FIGURE_SPACE, IDEOGRAPHIC_SPACE = "\u00a0", "\u202f", "\u2007", "\u3000"


@pytest.mark.parametrize("card", [
    "4111  1111  1111  1111",
    "4111   1111 1111    1111",
    "4111\t1111\t1111\t1111",
    f"4111{NBSP}1111{NBSP}1111{NBSP}1111",
    f"4111{NARROW_NBSP}1111{NARROW_NBSP}1111{NARROW_NBSP}1111",
    f"4111{FIGURE_SPACE}1111 {NBSP}1111\n1111",
])
def test_card_split_by_any_whitespace_run_is_masked(card):
    assert mask_pii(f"Card {card} charged") == "Card ****1111 charged"


def test_whitespace_runs_collapse_to_one_space():
    assert mask_pii(f"Crew\n\ncoffee{NBSP} run\t{IDEOGRAPHIC_SPACE}ok") == "Crew coffee run ok"


def test_fullwidth_card_number_is_masked():
    fullwidth = IDEOGRAPHIC_SPACE.join(["４１１１", "１１１１", "１１１１", "１１１１"])  # 4111 1111 1111 1111
    assert mask_pii(f"Card {fullwidth}") == "Card ****１１１１"


@pytest.mark.parametrize("text,masked", [
    ("Refund to josé@example.com", "Refund to [EMAIL]"),
    ("Zahlung an müller@bücher.de heute", "Zahlung an [EMAIL] heute"),
    ("contact 客服@例子.公司", "contact [EMAIL]"),
])
def test_unicode_email_is_masked(text, masked):
    assert mask_pii(text) == masked
