"""
eval/run_eval.py — score a categorizer prompt version against the gold set.

    ./venv/bin/python eval/run_eval.py --version v1
    ./venv/bin/python eval/run_eval.py --version v2
    ./venv/bin/python eval/run_eval.py --compare      # writes eval/RESULTS.md

Labeled gold rows are sent through the real categorizer path
(phase3_categorized.categorize: PII masking, reply validation and the one
repair retry), one request per business, as in production. Unlabeled rows
are skipped and counted.

Model replies are cached in eval/cache/<version>/, keyed by a hash of the
provider, the model id and the full prompt (which contains the rows), so a
re-run costs nothing and reproduces the same numbers.

Scoring:
- A gold label the version doesn't offer counts as Other for that version
  (v1 has no Travel/Transportation).
- A row that ends in NEEDS_REVIEW counts as incorrect; accuracy on answered
  rows is reported alongside, with review counts by reason.
- Per-category accuracy is recall per gold label. The confusion table is
  gold x predicted, with a REVIEW column. Under a version that offers
  Travel/Transportation, its precision, recall and F1 are reported.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import phase3_categorized as cat_mod
from afw.models import Category
from afw.prompt_versions import PROMPTS, prompt_version

EVAL_DIR = os.path.join(REPO, "eval")
DEFAULT_GOLD = os.path.join(EVAL_DIR, "gold_set.csv")
DEFAULT_CACHE = os.path.join(EVAL_DIR, "cache")
DEFAULT_RESULTS = os.path.join(EVAL_DIR, "results")
DEFAULT_RESULTS_MD = os.path.join(EVAL_DIR, "RESULTS.md")
TRAVEL = Category.TRAVEL_TRANSPORTATION.value
REVIEW = "REVIEW"
# Labels are written at the finest granularity any version offers.
VALID_LABELS = [c.value for c in PROMPTS["v2"].categories]


def model_identity():
    """(provider, model id) of the model call_model uses: part of the cache key."""
    import bedrock_client
    models = {"claude": bedrock_client.CLAUDE_MODEL_ID, "anthropic_direct": bedrock_client.ANTHROPIC_MODEL_ID}
    return bedrock_client.MODEL_PROVIDER, models.get(bedrock_client.MODEL_PROVIDER, bedrock_client.OPENAI_MODEL_ID)


def cache_key(identity, prompt):
    provider, model = identity
    return hashlib.sha256(f"{provider}\n{model}\n{prompt}".encode()).hexdigest()


def cached(call, identity, directory, stats):
    """Wrap a model call so each distinct prompt is answered once, ever."""
    def wrapped(prompt):
        path = os.path.join(directory, cache_key(identity, prompt) + ".txt")
        if os.path.exists(path):
            stats["cache_hits"] += 1
            with open(path, encoding="utf-8") as f:
                return f.read()
        stats["calls"] += 1
        reply = call(prompt)
        reply = reply if isinstance(reply, str) else ""
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(reply)
        return reply
    return wrapped


def load_gold(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labeled = [r for r in rows if r["label"].strip()]
    bad = [f"{r['id']}: {r['label']!r}" for r in labeled if r["label"].strip() not in VALID_LABELS]
    if bad:
        raise SystemExit(f"invalid labels (must be one of {', '.join(VALID_LABELS)}): {', '.join(bad)}")
    if not labeled:
        raise SystemExit(f"no labeled rows in {path}")
    return labeled, len(rows) - len(labeled)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate(gold_path, version, call, identity, cache_dir=DEFAULT_CACHE):
    offered = prompt_version(version).allowed
    labeled, unlabeled = load_gold(gold_path)
    with open(gold_path, "rb") as f:
        gold_sha256 = hashlib.sha256(f.read()).hexdigest()

    stats = Counter(calls=0, cache_hits=0)
    wrapped = cached(call, identity, os.path.join(cache_dir, version), stats)
    by_business = defaultdict(list)
    for r in labeled:
        by_business[r["business"]].append(r)

    predicted = {}
    for rows in by_business.values():
        sent = [{"id": r["id"], "merchant": r["merchant"], "description": r["description"],
                 "direction": "DEBIT", "is_refund": False} for r in rows]
        accepted, failures, _, _ = cat_mod.categorize(sent, version=version, call=wrapped)
        for r in rows:
            predicted[r["id"]] = accepted.get(r["id"]) or f"{REVIEW}:{failures[r['id']]}"

    scored = []
    for r in labeled:
        label = r["label"].strip()
        scored.append({"id": r["id"], "business": r["business"], "merchant": r["merchant"], "label": label,
                       "scored_label": label if label in offered else Category.OTHER.value,
                       "predicted": predicted[r["id"]]})

    correct = sum(1 for r in scored if r["predicted"] == r["scored_label"])
    answered = [r for r in scored if not r["predicted"].startswith(REVIEW)]
    order = [c.value for c in prompt_version(version).categories]
    per_category, confusion = {}, {}
    for category in order:
        rows = [r for r in scored if r["scored_label"] == category]
        if rows:
            hits = sum(1 for r in rows if r["predicted"] == category)
            per_category[category] = {"support": len(rows), "correct": hits, "accuracy": hits / len(rows)}
            confusion[category] = dict(Counter(r["predicted"].split(":")[0] for r in rows))

    travel = None
    if TRAVEL in offered:
        tp = sum(1 for r in scored if r["predicted"] == TRAVEL and r["scored_label"] == TRAVEL)
        n_predicted = sum(1 for r in scored if r["predicted"] == TRAVEL)
        n_gold = sum(1 for r in scored if r["scored_label"] == TRAVEL)
        precision, recall = _ratio(tp, n_predicted), _ratio(tp, n_gold)
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else None)
        travel = {"precision": precision, "recall": recall, "f1": f1,
                  "predicted": n_predicted, "gold": n_gold, "true_positive": tp}

    provider, model = identity
    return {
        "version": version, "provider": provider, "model": model, "gold_sha256": gold_sha256,
        "labeled": len(scored), "unlabeled": unlabeled,
        "correct": correct, "accuracy": correct / len(scored),
        "answered": len(answered),
        "answered_accuracy": _ratio(sum(1 for r in answered if r["predicted"] == r["scored_label"]), len(answered)),
        "review_counts": dict(Counter(r["predicted"].split(":", 1)[1] for r in scored if r not in answered)),
        "per_category": per_category, "confusion": confusion, "travel": travel,
        "calls": stats["calls"], "cache_hits": stats["cache_hits"],
        "rows": scored,
    }


# ------------------------------------------------------------------ RESULTS.md


def _pct(value):
    return "—" if value is None else f"{value * 100:.1f}%"


def results_markdown(v1, v2):
    if v1["gold_sha256"] != v2["gold_sha256"]:
        raise SystemExit("v1 and v2 were scored on different gold sets; re-run both on the same file")
    t1, t2 = v1["travel"] or {}, v2["travel"] or {}
    lines = [
        "# Categorizer eval: v1 vs v2",
        "",
        (f"- Gold set: `eval/gold_set.csv`, sha256 `{v2['gold_sha256']}`, {v2['labeled']} labeled purchases"
         f" ({v2['unlabeled']} unlabeled, skipped)."),
        f"- Model: {v1['provider']} / {v1['model']} (v1), {v2['provider']} / {v2['model']} (v2).",
        (f"- Results are a single run, pinned by cache (`eval/cache/`): re-running costs nothing and "
         f"reproduces these numbers; a fresh run against the model could differ. v1 generated "
         f"{v1.get('generated_at', 'n/a')}, v2 generated {v2.get('generated_at', 'n/a')}."),
        (f"- Scoring: v1 can't answer {TRAVEL}, so gold {TRAVEL} labels count as Other when scoring v1. "
         "Review rows count as incorrect."),
        "",
        "| Metric | v1 | v2 |",
        "|---|---|---|",
        f"| Overall accuracy | {_pct(v1['accuracy'])} | {_pct(v2['accuracy'])} |",
        f"| Accuracy on answered rows | {_pct(v1['answered_accuracy'])} | {_pct(v2['answered_accuracy'])} |",
        f"| Review rows | {v1['labeled'] - v1['answered']} | {v2['labeled'] - v2['answered']} |",
        f"| {TRAVEL} precision | {_pct(t1.get('precision'))} | {_pct(t2.get('precision'))} |",
        f"| {TRAVEL} recall | {_pct(t1.get('recall'))} | {_pct(t2.get('recall'))} |",
        f"| {TRAVEL} F1 | {_pct(t1.get('f1'))} | {_pct(t2.get('f1'))} |",
        "",
        "## Per-category accuracy (recall on gold labels)",
        "",
        "| Category | v1 | v2 |",
        "|---|---|---|",
    ]
    for category in VALID_LABELS:
        cells = []
        for result in (v1, v2):
            stats = result["per_category"].get(category)
            cells.append("—" if not stats else f"{_pct(stats['accuracy'])} ({stats['correct']}/{stats['support']})")
        lines.append(f"| {category} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------- CLI


def main(argv=None, call=None, cache_dir=DEFAULT_CACHE, results_dir=DEFAULT_RESULTS,
         results_md=DEFAULT_RESULTS_MD):
    parser = argparse.ArgumentParser(description="Score a categorizer prompt version against the gold set.")
    parser.add_argument("--version", choices=list(PROMPTS))
    parser.add_argument("--gold", default=DEFAULT_GOLD)
    parser.add_argument("--compare", action="store_true", help="write RESULTS.md from results/v1.json and v2.json")
    args = parser.parse_args(argv)

    if args.compare:
        results = []
        for version in ("v1", "v2"):
            with open(os.path.join(results_dir, f"{version}.json")) as f:
                results.append(json.load(f))
        with open(results_md, "w", encoding="utf-8") as f:
            f.write(results_markdown(*results))
        print(f"Wrote {results_md}")
        return
    if not args.version:
        parser.error("--version is required unless --compare is given")

    if call is None:
        import bedrock_client
        call = bedrock_client.call_model
    result = evaluate(args.gold, args.version, call, model_identity(), cache_dir)
    result["generated_at"] = datetime.now(timezone.utc).date().isoformat()
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{args.version}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"{args.version}: accuracy {_pct(result['accuracy'])} on {result['labeled']} labeled rows "
          f"({result['calls']} model calls, {result['cache_hits']} cache hits) -> {path}")


if __name__ == "__main__":
    main()
