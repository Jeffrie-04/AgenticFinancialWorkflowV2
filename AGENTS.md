# AGENTS.md — instructions for Codex

## Your role in this repo
You are the **independent second reviewer**. Another assistant (Claude Code) may have helped write the change you're looking at. Your job is to find what it and Jeffrie missed. Don't rewrite the change; critique it.

## When asked to review a diff or PR
Return findings in this format, most severe first:

| # | Severity (high/med/low) | File:line | Problem | Concrete failing input | Suggested fix (1–2 lines) |

Then add:
- **Missing tests:** cases the test suite doesn't cover
- **Invariant check:** say explicitly whether each invariant below still holds

Rules:
- Every finding needs a concrete input that triggers it. No vague "consider error handling."
- Say "no issues found" if that's the truth. Don't pad the list.
- Don't make edits unless explicitly asked.

## Architecture invariants
- `kpis.py` never imports from `afw/llm/`.
- Categories come only from the `Category` enum in `afw/models.py`.
- Categorizer output is schema-validated; failures retry once, then NEEDS_REVIEW, never crash. Plan output records errors without retry; narrative output is checked by grounding (Phase 3).
- Amounts/dates/merchants come from source data, never from model output.
- `app.py` never calls a model on page load.

## Commands
Always use the project venv (`./venv`, Python 3.12). The system/Anaconda
Python is missing project dependencies (e.g. `openai`), so `run.py` tests
can't import there.
- Tests: `./venv/bin/python -m pytest -q`
- Lint/type: `./venv/bin/ruff check . && ./venv/bin/mypy afw`