"""
One-off: convert the frozen tests/fixtures/snapshots/<business>/categorized.json
files from the old LLM output format ({date, merchant, amount, category} per
row) to the id-based format ({id, category}) that phase3_categorized.py now
produces.

Each old row is matched to its ingested source row with the key join exactly
as it stood at commit fdca13c (the last commit before the id join replaced
it), loaded from git history so this script keeps working after that code is
deleted. The conversion refuses to write anything unless the join is perfect:
every source row used, and zero join problems, unknown rows, duplicates or
invalid rows. The frozen kpis.json files are never touched.

Run from the repo root:  ./venv/bin/python scripts/convert_snapshot_categorized.py
"""
import json
import os
import subprocess
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from afw import ingest

KEY_JOIN_COMMIT = "fdca13c"
SNAPSHOTS = os.path.join(REPO, "tests", "fixtures", "snapshots")
BUSINESSES = ["landscaper", "law_firm", "restaurant"]


def load_key_join():
    src = subprocess.run(["git", "show", f"{KEY_JOIN_COMMIT}:phase3_kpisnoAI.py"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    module = types.ModuleType("phase3_kpisnoAI_key_join")
    exec(compile(src, f"phase3_kpisnoAI.py@{KEY_JOIN_COMMIT}", "exec"), module.__dict__)  # noqa: S102 - our own code at a pinned commit
    return module.join_categories


def convert(business, join_categories):
    folder = os.path.join(SNAPSHOTS, business)
    path = os.path.join(folder, "categorized.json")
    with open(path) as f:
        old = json.load(f)["categorized"]
    if all("id" in c for c in old):
        print(f"{business}: already converted, skipped")
        return

    csv_path = os.path.join(folder, "transactions.csv")
    result = ingest.ingest_file(csv_path, ingest.config_for(csv_path))
    ingested = [t.model_dump(mode="json") for t in result.rows]

    rows, report = join_categories(ingested, old)
    problems = {k: v for k, v in report.items()
                if k not in ("rows_in_kpis", "excluded_total") and v}
    assert not problems, f"{business}: key join problems {problems}"
    assert report["rows_in_kpis"] == len(ingested) == len(old), f"{business}: row counts differ"

    converted = [{"id": r["id"], "category": r["category"]} for r in rows]
    with open(path, "w") as f:
        json.dump({"categorized": converted}, f, indent=2)
        f.write("\n")
    print(f"{business}: converted {len(converted)} rows")


if __name__ == "__main__":
    join = load_key_join()
    for name in BUSINESSES:
        convert(name, join)
