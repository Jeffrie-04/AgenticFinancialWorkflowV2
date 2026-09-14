import json
import os

from bedrock_client import get_bedrock_client

# Load inputs
with open("outputs/categorized.json", "r") as f:
    categorized = json.load(f)

with open("outputs/kpis.json", "r") as f:
    kpis = json.load(f)

with open("outputs/summary.txt", "r") as f:
    summary = f.read()

with open("outputs/categorized.json", "r") as f:
    original_data = json.load(f)

# ----- INSERT YOUR FINAL REFLECTION PROMPT HERE -----
reflection_prompt = """
Role:
You are an analytical review agent responsible for evaluating the quality of the categorization, KPIs, and summary generated in previous workflow stages. Your goal is to inspect outputs, detect weaknesses, and propose improvements for the next iteration of the agentic workflow.

Input:
You will be given the outputs of earlier stages, including:
categorized.json
kpis.json
summary.txt
the original transaction dataset
Use these to identify issues, errors, and areas for improvement.

Steps:
Review all outputs and:
Identify incorrect or questionable transaction categories.
Find KPI errors, inconsistencies, or suspicious values.
Suggest improvements to categorization logic (e.g., additional rules, merchant mappings).
Suggest prompt modifications to improve KPI accuracy and validation.
Provide reasoning explaining why the model may have made these mistakes.

Expectations:
Your reflection must:
Point out at least two specific weaknesses or errors
Discuss misclassifications found in categorized.json
Identify KPI inconsistencies based on the dataset
Suggest new or refined categorization rules
Suggest prompt improvements for better KPI extraction
Present a clear, logical, model-driven reasoning process
Be written in plain text, not JSON

Narrowing (Plain Text Only):
Return a concise reflection in paragraph form that addresses all required points. No JSON, no lists, plain text only

"""

# Build full prompt with all data
final_prompt = f"""
{reflection_prompt}

ORIGINAL_TRANSACTIONS:
{json.dumps(original_data, indent=2)}

CATEGORIZED_JSON:
{json.dumps(categorized, indent=2)}

KPIS_JSON:
{json.dumps(kpis, indent=2)}

SUMMARY_TXT:
{summary}

Return ONLY plain text.
"""

# Configure Bedrock
bedrock = get_bedrock_client()

# Call Claude
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
                "content": final_prompt
            }
        ]
    })
)

# Extract response
response_body = json.loads(response["body"].read())
reflection_text = response_body["content"][0]["text"].strip()

# Save output
os.makedirs("outputs", exist_ok=True)
with open("outputs/reflection.txt", "w") as f:
    f.write(reflection_text)

print("Reflection saved to outputs/reflection.txt")