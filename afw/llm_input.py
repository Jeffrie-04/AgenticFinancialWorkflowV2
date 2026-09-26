"""
afw/llm_input.py — the only way transaction rows reach an LLM prompt.

Rows come from outputs/ingested.json and only status OK rows are returned:
REJECTED and NEEDS_REVIEW rows never leave the deterministic pipeline. Of
those, only model_rows() (non-refund DEBITs) are put in a prompt.
"""
import json
import os

from afw.guards.pii import mask_pii
from afw.models import Direction, Status

# Free-text fields that can carry PII; everything else (ids, amounts, dates,
# direction) is passed through untouched.
MASKED_FIELDS = ("merchant", "description")


def load_ok_rows(outputs_dir):
    with open(os.path.join(outputs_dir, "ingested.json")) as f:
        rows = json.load(f)["transactions"]
    return [t for t in rows if t["status"] == Status.OK.value]


def model_rows(rows):
    """The rows whose text may be sent to the categorizer and the plan
    sample: non-refund DEBITs. CREDITs are Income by rule and refunds take
    their original DEBIT's category, so neither needs the model. Client
    names are not fully local, though: by deliberate decision, the top
    client (a CREDIT merchant) still reaches the summary and reflection
    prompts through the KPIs, PII-masked but with the name itself intact."""
    return [t for t in rows if t["direction"] == Direction.DEBIT.value and not t["is_refund"]]


def prompt_row(row, fields):
    """One row, reduced to `fields`, as a prompt line with PII masked in its
    merchant and description text. Every row put in a prompt goes through here."""
    return prompt_line({f: mask_pii(row[f]) if f in MASKED_FIELDS and isinstance(row[f], str) else row[f]
                        for f in fields})


def prompt_line(fields):
    """One row as a single JSON line for a delimited prompt block. json.dumps
    escapes quotes and newlines; "<" is escaped too, so untrusted text (a
    merchant or description) can't close the block's tag early."""
    return json.dumps(fields, ensure_ascii=False).replace("<", "\\u003c")
