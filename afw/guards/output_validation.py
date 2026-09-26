"""
afw/guards/output_validation.py — turn an untrusted model reply into data.

Nothing here exits or raises on a bad reply: every failure comes back as a
short error string written by us, never echoing the reply text, so it can be
logged, recorded, or put in a repair prompt safely.
"""
import json

FENCE = "```"


def extract_json(text):
    """-> (dict, None) for a reply containing a JSON object, else (None, error).
    Tolerates markdown code fences and prose around the object: the outermost
    {...} is parsed."""
    if not isinstance(text, str):
        return None, "reply is not text"
    text = text.strip()
    if not text:
        return None, "reply is empty"

    if text.startswith(FENCE):
        text = text[len(FENCE):]
        text = text.removeprefix("json")
    text = text.removesuffix(FENCE).strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        return None, "no JSON object found in reply"
    try:
        return json.loads(text[start:end + 1]), None
    except json.JSONDecodeError as e:
        return None, f"reply is not valid JSON ({e.msg} at char {e.pos})"
