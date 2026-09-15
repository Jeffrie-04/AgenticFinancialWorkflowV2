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


def main():
    for name, module in PHASES:
        print(f"[run.py] Running: {name}")
        try:
            module.main()
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
    sys.exit(main())
