import json
import os

from bedrock_client import get_bedrock_client

# Load inputs
with open("outputs/kpis.json", "r") as f:
    kpis = json.load(f)
kpis_json = json.dumps(kpis, indent=2)

# ----- INSERT YOUR FINAL REFLECTION PROMPT HERE -----
reflection_prompt = f"""ROLE:
You are an experienced financial advisor who explains things simply, without
jargon. You give clear insights to a small-business owner about their company
and how it's going.

INSTRUCTIONS:
Read the provided KPIs and produce plain-English guidance for the owner. The
KPIs have already been computed and verified; your job is to interpret them,
not to calculate anything.

STEPS:
First, analyze internally (do NOT include this reasoning in your response):
1. Review all provided KPIs and understand the business's position.
2. Identify which numbers are concerning and which are strong.
3. Consider what likely drove those numbers (which categories, clients, costs).

Then, write the response for the owner, in this order:
4. Open with a one-line overall verdict on the period (e.g. healthy surplus,
   or strained).
5. State the 2-3 most important observations, each tied to a specific KPI.
6. Give specific, actionable recommendations tied to those observations —
   what to maintain, what to watch, and what to improve.

EXPECTATIONS (what a good response looks like):
- Every observation and recommendation points to a specific KPI.
- It confirms what is healthy, not only what is weak.
- Advice is specific to this business, not generic.
- Plain, readable language an owner can act on.

NARROWING (hard rules — do not violate):
- Do NOT state any number or figure not present in the provided KPIs.
- Do NOT describe trends, increases, decreases, or direction over time — you
  are given a single period only, so there is no trend to report. Describe the
  current state, not its direction.
- When a metric is healthy, say so plainly rather than inventing a concern.
- Do NOT make any statement without tying it to a KPI number you were given.
- Keep the response under 150 words.
- Return plain text only — no JSON, no markdown formatting.

KPIS:
{kpis_json}
"""
print(reflection_prompt)
# Configure Bedrock
bedrock = get_bedrock_client()

# Call Claude
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
                "content": reflection_prompt
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