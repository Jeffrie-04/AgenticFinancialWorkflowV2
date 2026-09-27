"""
afw/retry.py — the one retry policy for model calls (bedrock_client.call_model).

Only temporary errors are retried; which errors are temporary, and any
Retry-After the provider sent, is decided by the caller (it knows the SDK).
This module imports no SDK and no LLM code.

- At most MAX_RETRIES retries, waiting about 1s, 2s, 4s plus up to 1s of
  random jitter.
- A Retry-After longer than the backoff is waited instead of it.
- Never more than MAX_TOTAL_WAIT seconds of waiting in total: if the next
  wait would go over, it gives up at once rather than retrying before the
  provider said to.
- After the last attempt the original exception is re-raised unchanged, so
  a real outage still stops run.py with exit 1.

Each retry is logged to stderr with the attempt, error type, HTTP status and
wait. The exception message is never logged (a provider error body can echo
the request), and neither is the prompt.
"""
import random
import sys
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

MAX_RETRIES = 3
MAX_TOTAL_WAIT = 30.0


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _log_stderr(message: str) -> None:
    print(message, file=sys.stderr)


def _describe(exc: BaseException, status_of: Callable[[BaseException], int | None]) -> str:
    status = status_of(exc)
    return type(exc).__name__ + ("" if status is None else f" (status {status})")


def with_retries(
    fn: Callable[[], T],
    *,
    is_transient: Callable[[BaseException], bool],
    retry_after: Callable[[BaseException], float | None],
    status_of: Callable[[BaseException], int | None] = _status_code,
    sleep: Callable[[float], object] = time.sleep,
    rand: Callable[[], float] = random.random,
    log: Callable[[str], None] = _log_stderr,
) -> T:
    """Call fn(), retrying temporary errors under the policy above."""
    waited = 0.0
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            return fn()
        except Exception as exc:
            if attempt > MAX_RETRIES or not is_transient(exc):
                raise
            wait = 2 ** (attempt - 1) + rand()
            hint = retry_after(exc)
            if hint is not None and hint > wait:
                wait = hint
            if waited + wait > MAX_TOTAL_WAIT:
                log(f"[call_model] giving up after {_describe(exc, status_of)}: waiting {wait:.1f}s "
                    f"would exceed the {MAX_TOTAL_WAIT:.0f}s retry budget ({waited:.1f}s used)")
                raise
            log(f"[call_model] retry {attempt}/{MAX_RETRIES} after {_describe(exc, status_of)}; "
                f"waiting {wait:.1f}s")
            sleep(wait)
            waited += wait
    raise AssertionError("unreachable: the last attempt returns or raises")
