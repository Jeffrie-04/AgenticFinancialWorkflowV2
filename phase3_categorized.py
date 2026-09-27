"""
phase3_categorized.py — LLM categorization of validated transactions.

INPUT : outputs/ingested.json. Only OK non-refund DEBITs are sent to the
        model; REJECTED and NEEDS_REVIEW rows never appear in a prompt.
        OK non-refund CREDITs are income by rule; refunds take their original
        DEBIT's category in the KPI join, so neither is sent.
OUTPUT: outputs/categorized.json
        {"categorized": [{"id", "category"}],   validated and rule-assigned
         "review": [{"id", "reason"}],          rows the reply failed for
         "llm": {...counts}}

The model sees id, merchant, description and direction per row (no amounts,
no dates) and returns only a category per id. The reply is untrusted: it is
validated by afw/guards/output_validation.check_reply, and every row it fails
for goes to "review" with a reason (NEEDS_REVIEW), never guessed.
"""
import json
import os
from collections import Counter

from afw.guards.output_validation import check_reply, extract_json
from afw.llm_input import load_ok_rows, model_rows, prompt_row
from afw.models import Category, Direction
from afw.prompt_versions import PRODUCTION_VERSION, prompt_version
from bedrock_client import call_model

ROWS_START = "<transactions>"
ROWS_END = "</transactions>"
LLM_COUNTS = ("calls", "retried_ids", "invalid_items", "unknown_ids", "duplicate_ids",
              "unparseable_replies", "rule_assigned", "refunds_not_sent")

# What a repair prompt says about each failing id. The text is always ours:
# nothing the model wrote is ever echoed back into a prompt.
PROBLEM_TEXT = {
    "llm_missing": "missing from your reply",
    "llm_conflict": "returned more than once with different categories",
    "category_direction_mismatch": f"a DEBIT (money out) cannot be {Category.INCOME.value}",
}


def problem_text(reason, version=PRODUCTION_VERSION):
    if reason == "llm_invalid_category":
        names = ", ".join(c.value for c in prompt_version(version).categories)
        return f"category is not one of: {names}"
    return PROBLEM_TEXT[reason]


def build_prompt(rows, rejection=None, version=PRODUCTION_VERSION):
    """The categorizer prompt for `rows`, from the versioned file in prompts/.
    A retry uses the same builder, so the repair prompt has the same rules,
    masking and delimiters, plus the `rejection` section explaining what was
    wrong."""
    block = "\n".join(prompt_row(t, ("id", "merchant", "description", "direction")) for t in rows)
    prompt = prompt_version(version).render(block)
    if rejection:
        prompt += f"""

YOUR PREVIOUS REPLY WAS REJECTED.
{rejection}
Reply again with ONLY the JSON object: one entry per id in the block, ids copied exactly."""
    return prompt


def describe_failures(check, version=PRODUCTION_VERSION):
    """The rejection section for ids that failed validation."""
    lines = [f"- {row_id}: {problem_text(reason, version)}" for row_id, reason in check.failures.items()]
    return "These ids had problems:\n" + "\n".join(lines)


def ask(rows, rejection, version, call):
    """One model call for `rows` -> (ReplyCheck or None, error or None)."""
    obj, error = extract_json(call(build_prompt(rows, rejection, version)))
    if error:
        return None, error
    check = check_reply(obj, {t["id"]: t for t in rows}, prompt_version(version).allowed)
    return (None, check.error) if check.error else (check, None)


def categorize(rows, version=PRODUCTION_VERSION, call=None):
    """At most two model calls: the request, and one repair retry if needed.
    -> (accepted {id: category}, failures {id: reason}, counts, last error).
    If the whole reply is unusable, the retry resends every row; if only some
    ids fail, it resends just those, and accepted rows are kept as they are.
    `call` defaults to the production model (the eval harness injects its own)."""
    call = call or call_model
    counts = Counter(calls=1)
    check, error = ask(rows, None, version, call)
    if error:
        counts.update(unparseable_replies=1, calls=1, retried_ids=len(rows))
        check, error = ask(rows, f"It could not be used: {error}.", version, call)
        if error:
            counts["unparseable_replies"] += 1
            return {}, {t["id"]: "llm_unparseable" for t in rows}, counts, error
        counts.update(check.counts)
        return check.accepted, check.failures, counts, None

    counts.update(check.counts)
    if not check.failures:
        return check.accepted, {}, counts, None

    failing = [t for t in rows if t["id"] in check.failures]
    counts.update(calls=1, retried_ids=len(failing))
    retry, retry_error = ask(failing, describe_failures(check, version), version, call)
    if retry_error:
        counts["unparseable_replies"] += 1
        return check.accepted, check.failures, counts, None  # first-round reasons stand
    counts.update(retry.counts)
    return {**check.accepted, **retry.accepted}, retry.failures, counts, None


def main(outputs_dir="outputs"):
    rows = load_ok_rows(outputs_dir)
    to_model = model_rows(rows)
    counts = Counter(rule_assigned=sum(1 for t in rows if t["direction"] == Direction.CREDIT.value
                                       and not t["is_refund"]),
                     refunds_not_sent=sum(1 for t in rows if t["is_refund"]))
    accepted, failures, error = {}, {}, None
    if to_model:
        accepted, failures, model_counts, error = categorize(to_model)
        counts.update(model_counts)

    categorized, review = [], []
    for t in rows:
        if t["is_refund"]:
            continue
        if t["direction"] == Direction.CREDIT.value:
            categorized.append({"id": t["id"], "category": Category.INCOME.value})
        elif t["id"] in accepted:
            categorized.append({"id": t["id"], "category": accepted[t["id"]]})
        else:
            review.append({"id": t["id"], "reason": failures[t["id"]]})
    print(f"Categorized: {len(categorized)} rows ({counts['rule_assigned']} by rule), "
          f"{len(review)} to review" + (f" ({error})" if error else ""))

    # Nothing from the model is kept except validated {id, category}; failures
    # are recorded per row, never raised, so nothing crashes and nothing is dropped.
    result = {"categorized": categorized, "review": review,
              "llm": {k: counts[k] for k in LLM_COUNTS}, **({"error": error} if error else {})}
    with open(os.path.join(outputs_dir, "categorized.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
