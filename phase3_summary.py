import json
import os

from afw.guards.pii import mask_strings
from bedrock_client import call_model


def main(outputs_dir="outputs"):
    with open(os.path.join(outputs_dir, "kpis.json"), "r") as f:
        kpis = json.load(f)

    # -------------------------------
    # INSERT YOUR SUMMARIZATION PROMPT HERE
    # -------------------------------
    prompt = f"""Role:
You are a financial reporting agent. You write a short, neutral monthly
financial recap for a small-business owner, based on already-computed KPIs.

Task:
Using ONLY the KPIs provided below, write a single plain-text paragraph
(≤100 words) that recaps the month. This is a factual recap, not advice —
state what the numbers show, not what the owner should do.

Cover, in a natural flow:
- Total income and total spend, and the net cash flow (surplus or deficit).
- The largest spending categories (from spend_by_category).
- Any notable concentration — e.g. how much of income comes from the top
  clients (from income_concentration).

Rules:
- Use ONLY numbers present in the KPIs below. Do not invent or recompute any figure.
- Do NOT give recommendations or advice — that is a separate step.
- Do NOT describe trends or changes over time — this is a single period.
- Plain text only: one paragraph, no lists, no JSON, no markdown, ≤100 words.

KPIS:
{json.dumps(mask_strings(kpis), indent=2)}
"""


    summary_text = call_model(prompt).strip()

    # Save to outputs/summary.txt
    os.makedirs(outputs_dir, exist_ok=True)

    with open(os.path.join(outputs_dir, "summary.txt"), "w") as f:
        f.write(summary_text)


if __name__ == "__main__":
    main()
