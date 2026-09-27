import json
import os

from afw.guards.grounding import display_kpis, fallback_reflection
from afw.guards.pii import mask_strings
from afw.narrative import COPY_EXACTLY, grounded_text, record_grounding
from bedrock_client import call_model


def main(outputs_dir="outputs"):
    # Load inputs
    with open(os.path.join(outputs_dir, "kpis.json"), "r") as f:
        kpis = json.load(f)
    # Prompt copy only: display-formatted numbers and PII-masked names; kpis.json is unchanged.
    kpis_json = json.dumps(mask_strings(display_kpis(kpis)), indent=2)

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
- Review all provided KPIs and understand the business's position.
- Identify which numbers are concerning and which are strong.
- Consider what likely drove those numbers (which categories, clients, costs).

Then, write the response for the owner, in this order:
- Open with a one-line overall verdict on the period (e.g. healthy surplus,
  or strained).
- State the most important observations, each tied to a specific KPI.
- Give specific, actionable recommendations tied to those observations —
  what to maintain, what to watch, and what to improve.

EXPECTATIONS (what a good response looks like):
- Every observation and recommendation points to a specific KPI.
- It confirms what is healthy, not only what is weak.
- Advice is specific to this business, not generic.
- Plain, readable language an owner can act on.

NARROWING (hard rules — do not violate):
- Do NOT state any number or figure not present in the provided KPIs.
- {COPY_EXACTLY}
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
    #Test if prompt was fully built correctly
    #print(reflection_prompt)

    # The text is checked against the KPIs: regenerated once if a number isn't
    # a KPI value, then replaced by a deterministic fallback.
    reflection_text, outcome, rejected = grounded_text(reflection_prompt, kpis["kpis"], fallback_reflection, call_model)
    record_grounding(outputs_dir, "reflection", outcome, rejected)

    # Save output
    os.makedirs(outputs_dir, exist_ok=True)
    reflection_path = os.path.join(outputs_dir, "reflection.txt")
    with open(reflection_path, "w") as f:
        f.write(reflection_text)

    print(f"Reflection saved to {reflection_path}")


if __name__ == "__main__":
    main()
