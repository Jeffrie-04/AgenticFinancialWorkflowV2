"""
afw/guards/output_validation.py — turn an untrusted model reply into data.

Nothing here exits or raises on a bad reply: every failure comes back as a
short error string written by us, never echoing the reply text, so it can be
logged, recorded, or put in a repair prompt safely.
"""
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError

from afw.models import Category, direction_allows

FENCE = "```"
# Far above any real categorizer reply (about 60 characters per row); a longer
# reply is treated as unusable rather than parsed.
MAX_REPLY_CHARS = 100_000
CATEGORY_VALUES = {c.value for c in Category}


def extract_json(text):
    """-> (dict, None) for a reply containing a JSON object, else (None, error).
    Tolerates markdown code fences and prose around the object: the outermost
    {...} is parsed."""
    if not isinstance(text, str):
        return None, "reply is not text"
    text = text.strip()
    if not text:
        return None, "reply is empty"
    if len(text) > MAX_REPLY_CHARS:
        return None, f"reply is too long ({len(text)} characters)"

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
    except ValueError:  # e.g. an integer longer than Python's digit limit
        return None, "reply is not valid JSON (a value is out of range)"
    except RecursionError:
        return None, "reply is nested too deeply"


class ReplyEnvelope(BaseModel):
    """The reply's outer shape. Items are validated one by one, so one bad
    item never costs the valid ones."""
    model_config = ConfigDict(extra="ignore")
    categorized: list[Any]


class ReplyItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: StrictStr
    category: StrictStr


@dataclass
class ReplyCheck:
    accepted: dict = field(default_factory=dict)  # id -> category
    failures: dict = field(default_factory=dict)  # id -> reason
    counts: Counter = field(default_factory=Counter)
    error: str | None = None  # the reply as a whole was unusable


def check_reply(obj, sent):
    """Validate a parsed categorizer reply against the rows sent in this
    request (sent: id -> row with "direction"). Every sent id ends up in
    exactly one of accepted or failures:

    - llm_missing: no well-formed item for the id
    - llm_conflict: the id came back with different categories
    - llm_invalid_category: the category isn't a Category value
    - category_direction_mismatch: e.g. a DEBIT categorized as Income

    Malformed items (null, non-object, non-string id or category) and ids
    that weren't sent are ignored and counted; a repeat with the same
    category is accepted once and counted.
    """
    result = ReplyCheck()
    try:
        envelope = ReplyEnvelope.model_validate(obj)
    except ValidationError:
        result.error = "reply has no 'categorized' list"
        return result

    returned = defaultdict(list)
    for raw in envelope.categorized:
        try:
            item = ReplyItem.model_validate(raw)
        except ValidationError:
            result.counts["invalid_items"] += 1
            continue
        if item.id not in sent:
            result.counts["unknown_ids"] += 1
            continue
        returned[item.id].append(item.category)

    for row_id, row in sent.items():
        categories = returned.get(row_id)
        if not categories:
            result.failures[row_id] = "llm_missing"
        elif len(set(categories)) > 1:
            result.failures[row_id] = "llm_conflict"
        elif categories[0] not in CATEGORY_VALUES:
            result.failures[row_id] = "llm_invalid_category"
        elif not direction_allows(row["direction"], categories[0]):
            result.failures[row_id] = "category_direction_mismatch"
        else:
            result.accepted[row_id] = categories[0]
            result.counts["duplicate_ids"] += len(categories) - 1
    return result
