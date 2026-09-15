import json
import os

from bedrock_client import get_bedrock_client


def main():
    with open("outputs/categorized.json", "r") as f:
        categorized = json.load(f)

    with open("outputs/kpis.json", "r") as f:
        kpis = json.load(f)

    # -------------------------------
    # INSERT YOUR SUMMARIZATION PROMPT HERE
    # -------------------------------
    prompt = f"""
Role:
You are a financial summary generation agent with expertise in interpreting categorized transactions and computed KPIs. Your goal is to produce a clear, concise, and professional monthly financial summary.

Input:
You will be given:
Categorized transactions
KPI results (total spend, total income, top merchants, average expense)
Use this information to generate a short financial summary ≤100 words.

Steps:
Review the categorized transactions and KPIs.
Identify major spending categories.
Note total spend and total income.
Recognize top merchants with significant spending.
Form a concise, readable, and professional financial overview of the month.

Expectations:
The summary must:
Highlight major spending categories
Mention total income and total spend
Point out top merchants
Provide meaningful insight into overall financial health
Be clear, concise, and ≤100 words
Contain only plain text (no JSON, no formatting, no bullet points)

Narrowing (Plain Text Only):
Return a single short financial summary paragraph (≤100 words). Nothing else.

CATEGORIZED_TRANSACTIONS:
{json.dumps(categorized, indent=2)}

KPIS:
{json.dumps(kpis, indent=2)}

Remember: Return ONLY plain text. No JSON, no formatting.
"""


    bedrock = get_bedrock_client()
    # Call Claude Haiku since Titan was having issues reading the large input
    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
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
