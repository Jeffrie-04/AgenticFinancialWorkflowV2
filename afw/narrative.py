"""
afw/narrative.py — grounded narrative text (summary and advisor reflection).

The model's text is checked by afw/guards/grounding.check_grounding. If a
number isn't a KPI value, the text is regenerated once with a fixed message
naming the unsupported numbers (our parsed tokens, never the model's
sentences). If it still fails, a deterministic fallback built from the KPIs
replaces it. The outcome is recorded in data_quality.json["grounding"].

Every rejected attempt is saved to outputs/narrative_attempts.json for local
debugging only: the file is git-ignored and never read back into a prompt.
"""
import json
import os

from afw.guards.grounding import check_grounding
from afw.guards.pii import mask_pii, mask_strings

# Added to the summary and reflection prompts.
COPY_EXACTLY = "Copy numbers and names exactly as they appear in the KPIs; don't round, sum, compute or shorten."
REGENERATE = ("Your previous text used numbers that are not in the KPIs: {numbers}. Rewrite it using only "
              "numbers copied exactly from the KPIs; do not round, add or compute.")


def _check(reply, prompt_kpis):
    """-> (text, unsupported tokens, ok). An empty or non-text reply fails."""
    text = reply.strip() if isinstance(reply, str) else ""
    result = check_grounding(text, prompt_kpis)
    return text, result.unsupported, bool(text) and result.ok


def grounded_text(prompt, kpis, fallback, call):
    """-> (text, outcome, rejected). `kpis` is the KPI dict (the inner "kpis"
    object); `fallback(kpis)` builds the deterministic text; `call` is the
    model. `rejected` lists each failed attempt as {attempt, text, unsupported}.

    Grounding checks the text against the KPIs *as the prompt showed them*
    (PII-masked names), since that is what the model could copy."""
    prompt_kpis = mask_strings(kpis)
    text, unsupported, ok = _check(call(prompt), prompt_kpis)
    if ok:
        return text, {"status": "grounded", "attempts": 1, "unsupported": []}, []
    rejected = [{"attempt": 1, "text": text, "unsupported": unsupported}]

    numbers = ", ".join(mask_pii(token) for token in unsupported) or "none (the reply was empty)"
    retry = f"{prompt}\n\n{REGENERATE.format(numbers=numbers)}"
    text, retry_unsupported, ok = _check(call(retry), prompt_kpis)
    if ok:
        return text, {"status": "regenerated", "attempts": 2, "unsupported": unsupported}, rejected
    rejected.append({"attempt": 2, "text": text, "unsupported": retry_unsupported})
    return fallback(kpis), {"status": "fallback", "attempts": 2, "unsupported": retry_unsupported}, rejected


def _update_json(path, update):
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    update(data)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def record_grounding(outputs_dir, kind, outcome, rejected):
    """Store one narrative's outcome under data_quality.json["grounding"][kind]
    (keeping the file's other sections), and its rejected attempts under
    narrative_attempts.json[kind]; a grounded run clears stale attempts."""
    def set_outcome(data):
        data.setdefault("grounding", {})[kind] = outcome

    def set_attempts(data):
        data[kind] = rejected

    _update_json(os.path.join(outputs_dir, "data_quality.json"), set_outcome)
    _update_json(os.path.join(outputs_dir, "narrative_attempts.json"), set_attempts)
