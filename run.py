import argparse
import os
import sys

import phase2_plan
import phase3_categorized
import phase3_kpisnoAI
import phase3_summary
import phase3_reflection

PHASES = [
    ("Phase 2 - Plan", phase2_plan),
    ("Phase 3 - Categorize", phase3_categorized),
    ("Phase 3 - KPIs (no AI)", phase3_kpisnoAI),
    ("Phase 3 - Summary", phase3_summary),
    ("Phase 3 - Reflection", phase3_reflection),
]

NEEDS_CSV = {phase2_plan, phase3_categorized}


def main(business_dir=None):
    if business_dir is not None:
        csv_path = os.path.join(business_dir, "transactions.csv")
        outputs_dir = os.path.join(business_dir, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)

    for name, module in PHASES:
        print(f"[run.py] Running: {name}")
        try:
            if business_dir is None:
                module.main()
            elif module in NEEDS_CSV:
                module.main(csv_path=csv_path, outputs_dir=outputs_dir)
            else:
                module.main(outputs_dir=outputs_dir)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            print(f"[run.py] FAILED at {name}: script exited (code={code}). "
                  f"See error output above for details. Stopping pipeline.")
            return code or 1
        except Exception as e:
            print(f"[run.py] FAILED at {name}: {type(e).__name__}: {e}. Stopping pipeline.")
            return 1
        print(f"[run.py] Completed: {name}")
    print("[run.py] Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--business-dir", default=None,
                         help="Run against businesses/<name>/ instead of the legacy data/outputs paths.")
    args = parser.parse_args()
    sys.exit(main(business_dir=args.business_dir))
