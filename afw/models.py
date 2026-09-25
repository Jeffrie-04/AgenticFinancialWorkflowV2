"""
afw/models.py — the transaction contract every ingested row must satisfy.

A row is never silently dropped: it is OK, NEEDS_REVIEW (usable, but a human
should look), or REJECTED (can't be used), always with a reason when not OK.
Amounts are Decimal, always > 0 with exactly 2 decimal places; the sign lives
in `direction`, converted from the bank's convention at ingest.
"""
import re
from datetime import date as Date
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DESCRIPTION_MAX = 500

# strftime directives a source's date_format may use, and the exact shape each
# must have in the input. Shape is checked before strptime so "2024-10-06" in a
# MM-DD-YYYY source is date_format (wrong layout), not date_invalid (Feb 30).
DATE_DIRECTIVES = {"%m": r"\d{2}", "%d": r"\d{2}", "%Y": r"\d{4}", "%y": r"\d{2}"}


class Direction(str, Enum):
    DEBIT = "DEBIT"    # money out
    CREDIT = "CREDIT"  # money in


class Status(str, Enum):
    OK = "OK"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    REJECTED = "REJECTED"


class SignConvention(str, Enum):
    NEGATIVE_IS_CREDIT = "negative_is_credit"  # card-statement style: -6500.0 = money in
    POSITIVE_IS_CREDIT = "positive_is_credit"  # bank-account style: +6500.0 = money in


def date_format_regex(fmt):
    """Translate a supported date_format into a full-match regex."""
    pattern, i = "", 0
    while i < len(fmt):
        token = fmt[i:i + 2]
        if token in DATE_DIRECTIVES:
            pattern += DATE_DIRECTIVES[token]
            i += 2
        elif fmt[i] == "%":
            raise ValueError(f"unsupported date directive {token!r} in {fmt!r}")
        else:
            pattern += re.escape(fmt[i])
            i += 1
    return re.compile(pattern)


class SourceConfig(BaseModel):
    """How to read one bank's export. Set per source; never guessed from data."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    account_id: str = "default"
    sign_convention: SignConvention = SignConvention.NEGATIVE_IS_CREDIT
    date_format: str = "%m-%d-%Y"
    currency: str = "USD"

    @field_validator("date_format")
    @classmethod
    def _supported_date_format(cls, v):
        date_format_regex(v)
        return v


class Transaction(BaseModel):
    id: str
    source_file: str
    source_row: int
    account_id: str
    date: Optional[Date] = None
    amount: Optional[Decimal] = None
    direction: Optional[Direction] = None
    merchant: str = ""
    description: str = Field("", max_length=DESCRIPTION_MAX)
    currency: str = "USD"
    is_transfer: bool = False
    is_refund: bool = False
    refund_of: Optional[str] = None
    status: Status
    reason: Optional[str] = None

    @model_validator(mode="after")
    def _check_contract(self):
        if self.status != Status.REJECTED:
            if self.date is None or self.direction is None or self.amount is None:
                raise ValueError("usable rows need date, direction and amount")
            if self.amount <= 0 or self.amount.as_tuple().exponent != -2:
                raise ValueError(f"amount must be > 0 with 2 decimal places, got {self.amount}")
        if (self.status == Status.OK) != (self.reason is None):
            raise ValueError("reason is required unless status is OK, and only then")
        if self.is_refund and (self.direction != Direction.CREDIT or self.refund_of is None):
            raise ValueError("a refund must be a CREDIT with refund_of set")
        return self
