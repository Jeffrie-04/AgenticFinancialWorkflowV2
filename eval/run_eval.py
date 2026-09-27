"""
eval/run_eval.py — score a categorizer prompt version against the gold set.

    ./venv/bin/python eval/run_eval.py --version v1
    ./venv/bin/python eval/run_eval.py --version v2
    ./venv/bin/python eval/run_eval.py --compare      # writes eval/RESULTS.md
    ./venv/bin/python eval/run_eval.py --version v2 --offline   # replay only (CI)

Labeled gold rows are sent through the real categorizer path
(phase3_categorized.categorize: PII masking, reply validation and the one
repair retry), one request per business, as in production. Unlabeled rows
are skipped and counted.

Model replies are cached in eval/cache/<version>/, keyed by a hash of the
provider, the model id, the (non-secret) endpoint and the full prompt, which
contains the rows. A re-run costs nothing and reproduces the same numbers.
API keys are never part of the key.

--offline replays a recorded run without any model, and any cache miss is
an error. The provider setting in the environment (MODEL_PROVIDER, base
URLs) may still be read when bedrock_client is imported, but it is never
used: the recorded identity and generated_at from results/<version>.json
are. CI uses it to check that
the committed cache still reproduces the committed results.

Scoring:
- A gold label the version doesn't offer counts as Other for that version
  (v1 has no Travel/Transportation).
- A row that ends in NEEDS_REVIEW counts as incorrect; accuracy on answered
  rows is reported alongside, with review counts by reason.
- Rows are identified by (business, id): content-hash ids can repeat across
  businesses. A duplicate id within one business is an error.
- Per-category accuracy is recall per gold label. The confusion table is
  gold x predicted, with a REVIEW column. Under a version that offers
  Travel/Transportation, its precision, recall and F1 = 2TP / (2TP + FP + FN)
  are reported: F1 is 0 when every Travel row is wrong, and None only when
  there is no Travel data at all.
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
    """(provider, model id, endpoint) of the model call_model uses: part of the
    cache key. The endpoint is the non-secret base URL the SDK reads from the
    environment; API keys are never read here."""
    import bedrock_client
    provider = bedrock_client.MODEL_PROVIDER
    if provider == "claude":
        return provider, bedrock_client.CLAUDE_MODEL_ID, "bedrock-runtime:us-east-1"
    if provider == "anthropic_direct":
        return provider, bedrock_client.ANTHROPIC_MODEL_ID, os.environ.get("ANTHROPIC_BASE_URL", "default")
    return provider, bedrock_client.OPENAI_MODEL_ID, os.environ.get("OPENAI_BASE_URL", "default")


def cache_key(identity, prompt):
    return hashlib.sha256("\n".join([*identity, prompt]).encode()).hexdigest()


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


def offline_call(version, identity):
    """The model call for --offline: every prompt must already be cached."""
    def miss(prompt):
        raise SystemExit(f"offline: cache miss for {version} ({cache_key(identity, prompt)[:12]}); "
                         "run without --offline to call the model")
    return miss


def load_gold(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labeled = [r for r in rows if r["label"].strip()]
    bad = [f"{r['id']}: {r['label']!r}" for r in labeled if r["label"].strip() not in VALID_LABELS]
    if bad:
        raise SystemExit(f"invalid labels (must be one of {', '.join(VALID_LABELS)}): {', '.join(bad)}")
    seen = Counter((r["business"], r["id"]) for r in rows)
    duplicates = [f"{b}/{i}" for (b, i), n in seen.items() if n > 1]
    if duplicates:
        raise SystemExit(f"duplicate ids within a business: {', '.join(duplicates)}")
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
            predicted[r["business"], r["id"]] = accepted.get(r["id"]) or f"{REVIEW}:{failures[r['id']]}"

    scored = []
    for r in labeled:
        label = r["label"].strip()
        scored.append({"id": r["id"], "business": r["business"], "merchant": r["merchant"], "label": label,
                       "scored_label": label if label in offered else Category.OTHER.value,
                       "predicted": predicted[r["business"], r["id"]]})

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
        false_positives, false_negatives = n_predicted - tp, n_gold - tp
        precision, recall = _ratio(tp, n_predicted), _ratio(tp, n_gold)
        f1 = _ratio(2 * tp, 2 * tp + false_positives + false_negatives)
        travel = {"precision": precision, "recall": recall, "f1": f1,
                  "predicted": n_predicted, "gold": n_gold, "true_positive": tp}

    provider, model, endpoint = identity
    return {
        "version": version, "provider": provider, "model": model, "endpoint": endpoint,
        "gold_sha256": gold_sha256,
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


def _comparison(v1, v2, level="##"):
    """The summary bullets, metrics table and per-category table for one v1/v2 pair."""
    if v1["gold_sha256"] != v2["gold_sha256"]:
        raise SystemExit("v1 and v2 were scored on different gold sets; re-run both on the same file")
    t1, t2 = v1["travel"] or {}, v2["travel"] or {}
    lines = [
        (f"- Gold set: `eval/gold_set.csv`, sha256 `{v2['gold_sha256']}`, {v2['labeled']} labeled purchases"
         f" ({v2['unlabeled']} unlabeled, skipped)."),
        (f"- Model: {v1['provider']} / {v1['model']} @ {v1['endpoint']} (v1), "
         f"{v2['provider']} / {v2['model']} @ {v2['endpoint']} (v2)."),
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
        f"{level} Per-category accuracy (recall on gold labels)",
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
    return lines


def results_markdown(v1, v2, before=None, note=None):
    """RESULTS.md: the current v1/v2 comparison; optionally the same scores
    with the original labels (`before`, a v1/v2 pair) and an audit note."""
    lines = ["# Categorizer eval: v1 vs v2", "", *_comparison(v1, v2)]
    if before:
        lines += ["", "## With the original labels", "",
                  "The same model replies, scored against the gold set as first labeled:", "",
                  *_comparison(*before, level="###")]
    if note:
        lines += ["", "## Label audit", "", note.strip()]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------- CLI


def main(argv=None, call=None, cache_dir=DEFAULT_CACHE, results_dir=DEFAULT_RESULTS,
         results_md=DEFAULT_RESULTS_MD):
    parser = argparse.ArgumentParser(description="Score a categorizer prompt version against the gold set.")
    parser.add_argument("--version", choices=list(PROMPTS))
    parser.add_argument("--gold", default=DEFAULT_GOLD)
    parser.add_argument("--compare", action="store_true", help="write RESULTS.md from results/v1.json and v2.json")
    parser.add_argument("--offline", action="store_true",
                        help="replay the recorded run from the cache only; any cache miss is an error")
    args = parser.parse_args(argv)

    if args.compare:
        def load_pair(directory):
            pair = []
            for version in ("v1", "v2"):
                with open(os.path.join(directory, f"{version}.json")) as f:
                    pair.append(json.load(f))
            return pair

        original = os.path.join(results_dir, "original_labels")
        before = load_pair(original) if os.path.exists(os.path.join(original, "v2.json")) else None
        note_path = os.path.join(results_dir, "label_audit.md")
        note = None
        if os.path.exists(note_path):
            with open(note_path, encoding="utf-8") as f:
                note = f.read()
        with open(results_md, "w", encoding="utf-8") as f:
            f.write(results_markdown(*load_pair(results_dir), before=before, note=note))
        print(f"Wrote {results_md}")
        return
    if not args.version:
        parser.error("--version is required unless --compare is given")

    path = os.path.join(results_dir, f"{args.version}.json")
    if args.offline:
        if not os.path.exists(path):
            raise SystemExit(f"offline: no recorded results to replay at {path}; run without --offline first")
        with open(path, encoding="utf-8") as f:
            recorded = json.load(f)
        identity = (recorded["provider"], recorded["model"], recorded["endpoint"])
        call = offline_call(args.version, identity)
        generated_at = recorded["generated_at"]  # when the cached replies were generated
    else:
        if call is None:
            import bedrock_client
            call = bedrock_client.call_model
        identity = model_identity()
        generated_at = datetime.now(timezone.utc).date().isoformat()
    result = evaluate(args.gold, args.version, call, identity, cache_dir)
    result["generated_at"] = generated_at
    os.makedirs(results_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"{args.version}: accuracy {_pct(result['accuracy'])} on {result['labeled']} labeled rows "
          f"({result['calls']} model calls, {result['cache_hits']} cache hits) -> {path}")


if __name__ == "__main__":
    main()
