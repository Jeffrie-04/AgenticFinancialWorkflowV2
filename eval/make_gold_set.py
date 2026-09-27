"""
eval/make_gold_set.py — write the unlabeled gold set for the categorizer eval.

Every purchase the categorizer would see, across the demo businesses: the
OK, non-refund DEBIT rows chosen by production's own rule (afw/ingest +
afw/llm_input.model_rows). Columns: id, business, merchant, description,
amount, label. The label column is left empty: labels are added by hand, and
this script never writes one. It refuses to overwrite a file that already
has any label, so hand labels can't be lost.

The text is the original, unmasked source text: this is a local labeling
copy, not a prompt.

Run from the repo root:  ./venv/bin/python eval/make_gold_set.py
"""
import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from afw import ingest
from afw.llm_input import model_rows

BUSINESSES = ["landscaper", "law_firm", "restaurant", "consultant"]
HEADER = ["id", "business", "merchant", "description", "amount", "label"]
DEFAULT_PATH = os.path.join(REPO, "eval", "gold_set.csv")


def gold_rows():
    rows = []
    for business in BUSINESSES:
        path = os.path.join(REPO, "businesses", business, "transactions.csv")
        ingested = [t.model_dump(mode="json") for t in ingest.ingest_file(path, ingest.config_for(path)).rows]
        ok = [t for t in ingested if t["status"] == "OK"]
        rows += [[t["id"], business, t["merchant"], t["description"], t["amount"], ""] for t in model_rows(ok)]
    return rows


def has_labels(path):
    if not os.path.exists(path):
        return False
    with open(path, newline="", encoding="utf-8") as f:
        return any(row.get("label", "").strip() for row in csv.DictReader(f))


def main(path=DEFAULT_PATH):
    if has_labels(path):
        raise SystemExit(f"{path} already has labels; refusing to overwrite hand-labeled data")
    rows = gold_rows()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} unlabeled purchases to {path}")


if __name__ == "__main__":
    main()
