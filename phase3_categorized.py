import json
import pandas as pd

from bedrock_client import get_bedrock_client, clean_json_text, parse_json_response

# Load transaction data
df = pd.read_csv('data/transactiondata.csv').head(5)

# RISEN Prompt
prompt = f"""ROLE: You are an expert financial transaction categorization agent for small business accounting.

INSTRUCTIONS: Categorize each transaction into exactly ONE category based on these rules:

**SHOPPING** (Retail & Supplies):
- Office supplies: Staples, Office Depot, Amazon Business
- Equipment: Best Buy, Dell, Apple Store
- Furniture: Home Depot, IKEA
- Bulk supplies: Costco Business

**DINING** (Food & Beverages):
- Coffee shops: Starbucks, Dunkin Donuts
- Restaurants: Chipotle, Panera Bread, Subway, McDonald's, Panda Express, Olive Garden
- Food delivery: Uber Eats, DoorDash
- Catering: Office Coffee Service

**UTILITIES** (Recurring Services & Bills):
- Software/SaaS: Adobe, Salesforce, QuickBooks, Slack, Zoom, Microsoft, Google Workspace, AWS, GitHub, Shopify, HubSpot, etc.
- Office rent: WeWork, office space
- Insurance: Business liability, professional insurance
- Bills: Verizon, PG&E, Water Company, Internet, Electric, Gas
- Payroll: Employee salaries, contractor payments
- Professional services: Legal, accounting

**INCOME** (Money Received - NEGATIVE amounts):
- Client payments: Invoice payments, retainer fees
- Project payments: Milestone payments, consulting fees
- Other income: Referral commissions, interest, deposits
- Rule: If amount is NEGATIVE, it's income

**OTHER** (Transportation, Travel, Miscellaneous):
- Transportation: Uber, Lyft, taxi
- Shipping: FedEx, UPS
- Fuel: Shell, Chevron, gas stations
- Travel: Hotels, airfare
- Entertainment: Movie theaters, events
- Banking fees: Stripe fees, transaction fees
- Waste services: Trash, recycling

STEPS TO CATEGORIZE:
1. Read the merchant name and description
2. Check if amount is negative (if yes → Income)
3. Match merchant to category rules above
4. If merchant matches multiple categories, use description to decide
5. Assign the most specific category

EXPECTATIONS - Output Format:
Return ONLY a valid JSON object with ALL {len(df)} transactions categorized.
Each transaction must have: date, merchant, amount, category

NARROWING - Critical Rules:
- NEGATIVE amounts are ALWAYS "Income" (e.g., -3500.00 = Income)
- SaaS subscriptions are "Utilities" not "Shopping"
- Client meetings at restaurants are "Dining"
- Office supplies from Amazon are "Shopping" not "Other"
- Payroll is "Utilities" not "Other"
- Gas stations are "Other" not "Utilities"

TRANSACTION DATA TO CATEGORIZE:
{df.to_csv(index=False)}

OUTPUT REQUIREMENTS:
Return ONLY valid JSON. No markdown. No explanations. No text before or after JSON.

Format:
{{
  "categorized": [
    {{"date": "2024-10-01", "merchant": "Example Corp", "amount": 100.00, "category": "Shopping"}},
    {{"date": "2024-10-02", "merchant": "Client ABC", "amount": -5000.00, "category": "Income"}}
  ]
}}
YOUR RESPONSE MUST START WITH {{ AND END WITH }}. Nothing else."""


# Configure timeout
bedrock = get_bedrock_client()

# Call Claude Haiku since Titan was having issues reading the large input
response = bedrock.invoke_model(
    modelId='us.anthropic.claude-haiku-4-5-20251001-v1:0',
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
text = clean_json_text(text)

# Parse JSON
categorized = parse_json_response(text)
print("JSON parsed successfully")

# Validate structure
if "categorized" not in categorized:
    print("Warning: Response missing 'categorized'")
    if isinstance(categorized, list):
        categorized = {"categorized": categorized}
    elif "items" in categorized:
        categorized = {"categorized": categorized["items"]}

# Validation
expected = len(df)
actual = len(categorized.get("categorized", []))
print(f"Categorized: {actual}/{expected} transactions")

if actual < expected:
    print(f"WARNING: Missing {expected - actual} transactions!")
print()

# Save
with open('outputs/categorized.json', 'w') as f:
    json.dump(categorized, f, indent=2)

