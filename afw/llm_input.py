"""
afw/llm_input.py — the only way transaction rows reach an LLM prompt.

Rows come from outputs/ingested.json and only status OK rows are returned:
REJECTED and NEEDS_REVIEW rows never leave the deterministic pipeline.
"""
import json
import os

from afw.models import Status


def load_ok_rows(outputs_dir):
    with open(os.path.join(outputs_dir, "ingested.json")) as f:
        rows = json.load(f)["transactions"]
    return [t for t in rows if t["status"] == Status.OK.value]


def prompt_line(fields):
    """One row as a single JSON line for a delimited prompt block. json.dumps
    escapes quotes and newlines; "<" is escaped too, so untrusted text (a
    merchant or description) can't close the block's tag early."""
    return json.dumps(fields, ensure_ascii=False).replace("<", "\\u003c")
