"""
afw/prompt_versions.py — categorizer prompts as versioned text files.

Each version is a file in prompts/ (a string.Template with one $block
placeholder for the delimited rows) plus the categories that version offers.
The category list is fixed per version, not derived from the Category enum,
so adding an enum value never changes an existing prompt. Tests check that
each file's "Allowed categories" line and guide lines match its list.

v2 is the production prompt, switched after the v1-vs-v2 eval (eval/RESULTS.md,
ADR 0004). v1 stays frozen and reproducible for comparison.
"""
import os
from dataclasses import dataclass
from functools import cache
from string import Template

from afw.models import Category

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")


@dataclass(frozen=True)
class PromptVersion:
    name: str
    filename: str
    categories: tuple  # the DEBIT categories this prompt offers, in prompt order

    @property
    def path(self):
        return os.path.join(PROMPTS_DIR, self.filename)

    @property
    def allowed(self):
        return {c.value for c in self.categories}

    def render(self, block):
        return _template(self.path).substitute(block=block)


@cache
def _template(path):
    with open(path, encoding="utf-8") as f:
        return Template(f.read())


PROMPTS = {
    "v1": PromptVersion("v1", "categorize_v1.txt",
                        (Category.UTILITIES, Category.SHOPPING, Category.DINING, Category.OTHER)),
    # v2 adds Travel/Transportation (hotels, fuel and EV charging, airfare,
    # rideshare, tolls, parking), which v1 files under Other. Evaluated
    # against v1 in eval/ (eval/RESULTS.md); production since that eval.
    "v2": PromptVersion("v2", "categorize_v2.txt",
                        (Category.UTILITIES, Category.SHOPPING, Category.DINING,
                         Category.TRAVEL_TRANSPORTATION, Category.OTHER)),
}
PRODUCTION_VERSION = "v2"


def prompt_version(name):
    if name not in PROMPTS:
        raise ValueError(f"unknown prompt version {name!r}; known: {', '.join(PROMPTS)}")
    return PROMPTS[name]
