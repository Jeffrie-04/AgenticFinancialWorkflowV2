# AgenticFinancialWorkflow

An agentic workflow that takes raw business transaction data (a CSV) and moves it through several stages to produce plain-English financial insights for small-business owners — metrics they can actually act on, derived from their own transactions.

## Why this exists

When people start a small business, they rarely have time to step back and look at the whole financial picture. There's so much on their plate that important signals in their own data get overlooked. This workflow does that stepping-back for them: it reads their transactions and surfaces the metrics — and the guidance — that tell them how the business is actually doing.

## Origin

This began as a team class project. The original assignment was to write prompts in different frameworks (RISEN, RAFT) to drive stages defined by our professor, and to connect those stages to Amazon Bedrock. My work was split evenly between the prompt engineering and building the Python scripts that wired the pipeline to Bedrock.

This repository is my individual continuation of that project. Since the class version, I've added a deterministic KPI engine with several new metrics, a test suite, an orchestrator to run the pipeline end-to-end, and an AI advisor stage; refined the prompts for better results; and moved the pipeline to a newer Claude model.

## Design principle: AI for language, Python for math

The core design decision is that **all financial math is computed in deterministic Python, and the LLM is only used for language tasks** — categorization, summaries, and advice. LLMs can hallucinate numbers, and a single made-up figure would undermine the whole point of a financial tool. Keeping the arithmetic in tested Python guarantees the numbers are correct; the model only ever *interprets* figures that were already computed and verified, never produces them.

## The pipeline

The workflow runs in five stages, each feeding the next:

1. **Plan** — establishes the analysis approach for the run.
2. **Categorize** — the LLM reads the raw transactions and assigns each to a category (e.g. Utilities, Dining, Income), so spending and income can be broken down meaningfully.
3. **KPIs** — deterministic Python computes all the metrics from the categorized data. No LLM involved. This is the source of truth.
4. **Summary** — the LLM writes a short plain-English overview of the period from the computed KPIs.
5. **Reflection (Advisor)** — the LLM reviews the KPIs, gives a direct verdict on how the business is doing that period, calls out the 2–3 most important observations tied to specific metrics, and offers actionable suggestions based on the business's situation.

## What it calculates

The engine produces six core metric groups: cash volume, net cash flow, categorical spend, income concentration, fixed vs. discretionary costs, and daily burn pace.

These matter because cash-flow mismanagement and sudden liquidity crunches cause most small-business failures. Net cash flow and burn pace reveal immediately whether operations are self-sustaining or bleeding cash; income concentration highlights dangerous over-reliance on a single client; and the category and fixed-cost breakdowns show exactly where overhead can be trimmed when margins tighten. Because these are computed in deterministic code rather than by a model, the arithmetic is exact — reliable financial signals, not guesswork.

## How I used AI

Two distinct ways. **Building it:** I used Claude to brainstorm the design, refine the prompts, and work through problems along the way — environment and dependency issues (boto3, Python setup), AWS Bedrock provisioning, and token-quota limits. **Inside the product:** the pipeline uses structured prompt frameworks (RAFT for planning, RISEN for the constrained tasks like categorization and the advisor), designed so the model stays grounded in the provided data and never invents figures.

## Tech stack

Python · AWS Bedrock (Claude Haiku 4.5) · boto3 · pandas · pytest · venv

## How to run it

```bash
# 1. Set up an isolated environment
python3 -m venv venv
source venv/bin/activate          # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure AWS credentials (Bedrock access required)
aws configure

# 4. Run the full pipeline
python run.py
```

Individual stages can also be run on their own (e.g. `python phase3_kpisnoAI.py`). Tests: `python -m pytest tests/ -v`.

## Limitations & roadmap

Current limitations: the analysis covers a single time period (no trend detection yet), "runway" can't be computed without a starting cash balance, the sample dataset is small, and input is currently a clean CSV rather than the bank/credit-card statements a real business would more likely have.

Next steps: **statement ingestion** — parsing real bank and credit-card statements into the transaction format the pipeline already expects, so the analysis works on what businesses actually have; **multi-month data** to unlock trend and period-over-period metrics; and a **UI** for uploading data and viewing results.