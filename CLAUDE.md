# CLAUDE.md

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
