import json
import pandas as pd

from bedrock_client import get_bedrock_client, clean_json_text, parse_json_response

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


# Configure timeout
bedrock = get_bedrock_client()

# Call Claude Haiku since Titan was having issues reading the large input
response = bedrock.invoke_model(
    modelId='anthropic.claude-3-haiku-20240307-v1:0',
    contentType='application/json',
    accept='application/json',
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8000,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
)

# Parse response
response_body = json.loads(response['body'].read())
text = response_body['content'][0]['text']


# JSON CLEANING
text = text.strip()

# Remove markdown code blocks
if text.startswith("```json"):
    text = text[7:]
if text.startswith("```"):
    text = text[3:]
if text.endswith("```"):
    text = text[:-3]
text = text.strip()

# Find JSON in response
start = text.find('{')
end = text.rfind('}') + 1
if start != -1 and end > start:
    text = text[start:end]
else:
    print("Error: Could not find JSON in response")
    print("Raw response:", text[:500])
    exit(1)

# Parse JSON
try:
    kpis = json.loads(text)
    print("JSON parsed successfully")
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
    print("Cleaned text:", text[:500])
    exit(1)

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