"""
afw/guards/pii.py — mask PII in text before it goes into a model prompt.

Applied only to merchant and description text (and to strings in the KPIs
sent to the summary and reflection prompts), never to amounts, dates or ids.
Person names are not masked: merchants are names. Local files keep the
original text; only prompts are masked.

Masks, in order:
- emails -> [EMAIL]
- US phone numbers written with separators or +1 -> [PHONE]
- card numbers: 13-19 digits, contiguous or in 4-digit groups (or Amex
  4-6-5) separated by one space or dash -> ****1234
- account numbers: contiguous runs of 8-17 digits -> ****1234

A number followed by a decimal part (an amount such as 12345678.90) is not a
card or account number. Long order or reference numbers are masked too;
that over-masking is accepted.
"""
import re

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<![\d+])(?:\+1[ .-]?)?(?:\(\d{3}\) ?|\d{3}[ .-])\d{3}[ .-]\d{4}(?!\d)")
NOT_AFTER_NUMBER = r"(?<![\d.,])"
NOT_BEFORE_NUMBER = r"(?!\d|[.,]\d)"
CARD = re.compile(NOT_AFTER_NUMBER
                  + r"(?:\d{4}([ -])\d{4}\1\d{4}\1\d{1,7}|\d{4}([ -])\d{6}\2\d{5}|\d{13,19})"
                  + NOT_BEFORE_NUMBER)
ACCOUNT = re.compile(NOT_AFTER_NUMBER + r"\d{8,17}" + NOT_BEFORE_NUMBER)


def _last_four(match):
    digits = re.sub(r"\D", "", match.group())
    return "****" + digits[-4:]


def mask_pii(text):
    text = EMAIL.sub("[EMAIL]", text)
    text = PHONE.sub("[PHONE]", text)
    text = CARD.sub(_last_four, text)
    return ACCOUNT.sub(_last_four, text)


def mask_strings(value):
    """mask_pii applied to every string inside nested dicts and lists (values
    only; keys are our own field names). Other values are returned as-is."""
    if isinstance(value, str):
        return mask_pii(value)
    if isinstance(value, dict):
        return {k: mask_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [mask_strings(v) for v in value]
    return value
