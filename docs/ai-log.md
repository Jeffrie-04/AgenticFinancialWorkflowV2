# AI Decision Log

Log an entry whenever an AI suggestion changes, or fails to change, what you ship. You write these, not the AI. Aim for 2–4 lines each. These entries are your interview stories.

**Verdict key:** ACCEPTED · MODIFIED · REJECTED · AI WAS WRONG · AI CAUGHT MY BUG

---

## Template

### YYYY-MM-DD · Phase N · short title
- **Tool:** Claude Code / Codex
- **I asked:**
- **It suggested:**
- **Verdict:**
- **Why / how I verified:** (test, docs, experiment, the eval number)

---

## Example (delete once you have real entries)

### 2026-09-25 · Phase 2 · retry strategy
- **Tool:** Claude Code
- **I asked:** How should the categorizer handle a malformed model response?
- **It suggested:** Retry up to 3 times with the same prompt.
- **Verdict:** MODIFIED
- **Why / how I verified:** At temperature 0 the same prompt tends to fail the same way again. I changed it to one retry with a *repair* prompt that includes the validation error, then NEEDS_REVIEW. The test `test_malformed_json_goes_to_review` covers it.