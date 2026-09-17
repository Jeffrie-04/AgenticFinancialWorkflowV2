import json
import pandas as pd

from bedrock_client import call_model, clean_json_text, parse_json_response


def main():
    # Load categorized transactions
    try:
        with open('outputs/categorized.json', 'r') as f:
            data = json.load(f)

        transactions = data.get('categorized', [])

    except FileNotFoundError:
        print("Error: outputs/categorized.json not found")
        print("Run phase3_categorized.py first!")
        exit(1)

    # RISEN Prompt
    prompt = f"""Role:
You are a financial data analysis agent with expertise in computing performance metrics from categorized financial transactions. You specialize in producing accurate KPI reports for small businesses.

Input:
You will be given a list of categorized transactions, each containing:
- date
- merchant
- amount
- category
You must use this dataset to compute financial KPIs and verify their accuracy.

Steps:
Identify all expense transactions (Shopping, Dining, Utilities, Other).
Calculate:
- total_spend = sum of all expense amounts
- total_income = sum of all Income amounts (amounts are negative; use their absolute values)
- top_merchants = 3 merchants with the highest total spending
- average_expense = total_spend ÷ number of expense transactions

Perform validation checks:
- total_spend > 0
- total_income aligns with Income entries
- top_merchants accurately reflect spending
- average_expense is mathematically correct

Expectations:
All calculations must be accurate and based strictly on the provided data.
Results must follow the exact JSON schema below.
All numeric values must be valid numbers (not strings).
top_merchants must contain exactly 3 merchants.
Your output must consist of ONLY valid JSON, no extra text, formatting, or commentary.

Narrowing (JSON Only):
Return your results in this exact structure:

CRITICAL: Numbers must NOT have commas. Use 6524.59 not 6,524.59

{{
  "kpis": {{
    "total_spend": 0,
    "total_income": 0,
    "top_merchants": ["Merchant1", "Merchant2", "Merchant3"],
    "average_expense": 0
  }}
}}
TRANSACTION DATA:
{json.dumps({"categorized": transactions}, indent=2)}


YOUR RESPONSE MUST START WITH {{ AND END WITH }}. Nothing else."""


    text = call_model(prompt)

    # JSON CLEANING
    text = clean_json_text(text)

    # Parse JSON
    kpis = parse_json_response(text)
    print("JSON parsed successfully")

    # Validate structure
    if "kpis" not in kpis:
        print("Warning: Response missing 'kpis'")
        if isinstance(kpis, list):
            kpis = {"kpis": kpis}
        elif "items" in kpis:
            kpis = {"kpis": kpis["items"]}



    # Save
    with open('outputs/kpisAI.json', 'w') as f:
        json.dump(kpis, f, indent=2)


if __name__ == "__main__":
    main()
