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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DESCRIPTION_MAX = 500
MAX_AMOUNT = Decimal("1000000000.00")

# strftime directives a source's date_format may use, and the exact shape each
# must have in the input. Shape is checked before strptime so "2024-10-06" in a
# MM-DD-YYYY source is date_format (wrong layout), not date_invalid (Feb 30).
DATE_DIRECTIVES = {"%m": r"\d{2}", "%d": r"\d{2}", "%Y": r"\d{4}", "%y": r"\d{2}"}


def normalize_merchant(name):
    """The one merchant-comparison key: stripped, internal whitespace
    collapsed, casefolded. Used for refund matching and the KPI category join;
    the stored `merchant` field is only stripped."""
    return " ".join(name.split()).casefold()


class Direction(str, Enum):
    DEBIT = "DEBIT"    # money out
    CREDIT = "CREDIT"  # money in


class Status(str, Enum):
    OK = "OK"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    REJECTED = "REJECTED"


class Category(str, Enum):
    """The only valid categories. LLM output is validated against this; an
    unknown value makes the row NEEDS_REVIEW, never a guessed category."""
    INCOME = "Income"
    UTILITIES = "Utilities"
    SHOPPING = "Shopping"
    DINING = "Dining"
    TRAVEL_TRANSPORTATION = "Travel/Transportation"
    OTHER = "Other"


# Only non-refund DEBITs are categorized by the model; non-refund CREDITs are
# Income by rule, and refunds take their original DEBIT's category. Which
# categories a prompt offers is fixed per prompt version (afw/prompt_versions.py).


def direction_allows(direction, category):
    """False when a category contradicts the row's direction: money out can't
    be Income."""
    return not (direction == Direction.DEBIT.value and category == Category.INCOME.value)


class SignConvention(str, Enum):
    NEGATIVE_IS_CREDIT = "negative_is_credit"  # card-statement style: -6500.0 = money in
    POSITIVE_IS_CREDIT = "positive_is_credit"  # bank-account style: +6500.0 = money in


def date_format_regex(fmt):
    """Translate a supported date_format into a full-match regex. The format
    must name exactly one year (%Y or %y), one month and one day."""
    pattern, used, i = "", [], 0
    while i < len(fmt):
        token = fmt[i:i + 2]
        if token in DATE_DIRECTIVES:
            pattern += DATE_DIRECTIVES[token]
            used.append("%Y" if token == "%y" else token)
            i += 2
        elif fmt[i] == "%":
            raise ValueError(f"unsupported date directive {token!r} in {fmt!r}")
        else:
            pattern += re.escape(fmt[i])
            i += 1
    if sorted(used) != ["%Y", "%d", "%m"]:
        raise ValueError(f"date_format {fmt!r} must contain exactly one year, month and day")
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
    date: Date | None = None
    amount: Decimal | None = None
    direction: Direction | None = None
    merchant: str = ""
    description: str = Field("", max_length=DESCRIPTION_MAX)
    currency: str = "USD"
    is_transfer: bool = False
    is_refund: bool = False
    refund_of: str | None = None
    status: Status
    reason: str | None = None

    @model_validator(mode="after")
    def _check_contract(self):
        if self.status != Status.REJECTED:
            if self.date is None or self.direction is None or self.amount is None:
                raise ValueError("usable rows need date, direction and amount")
            if not 0 < self.amount <= MAX_AMOUNT or self.amount.as_tuple().exponent != -2:
                raise ValueError(f"amount must be in (0, {MAX_AMOUNT}] with 2 decimal places, "
                                 f"got {self.amount}")
            if self.currency != "USD":
                raise ValueError(f"usable rows must be USD, got {self.currency}")
        if self.status == Status.OK and not self.merchant.strip():
            raise ValueError("an OK row needs a merchant")
        if (self.status == Status.OK) != (self.reason is None):
            raise ValueError("reason is required unless status is OK, and only then")
        if self.is_refund and (self.direction != Direction.CREDIT or self.refund_of is None):
            raise ValueError("a refund must be a CREDIT with refund_of set")
        return self
