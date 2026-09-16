import json
import os

from bedrock_client import get_bedrock_client


def main():
    with open("outputs/kpis.json", "r") as f:
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
{json.dumps(kpis, indent=2)}
"""


    bedrock = get_bedrock_client()
    # Call Claude Haiku since Titan was having issues reading the large input
    response = bedrock.invoke_model(
        modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
    )

    # Extract text
    response_body = json.loads(response["body"].read())
    summary_text = response_body["content"][0]["text"].strip()

    # Save to outputs/summary.txt
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/summary.txt", "w") as f:
        f.write(summary_text)


if __name__ == "__main__":
    main()
