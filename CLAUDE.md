# CLAUDE.md

## Commands
Always use the project venv (`./venv`, Python 3.12). The system/Anaconda
Python is missing project dependencies (e.g. `openai`), so `run.py` tests
can't import there.
- Tests: `./venv/bin/python -m pytest -q`
- Lint/type: `./venv/bin/ruff check . && ./venv/bin/mypy afw`
