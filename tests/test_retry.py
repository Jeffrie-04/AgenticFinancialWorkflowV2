"""
tests/test_retry.py — the retry policy in afw/retry.py, provider-agnostic.
Exceptions are fakes; sleep and random are injected, so nothing sleeps.
"""
import pytest

from afw.retry import MAX_RETRIES, MAX_TOTAL_WAIT, with_retries


class Temporary(Exception):
    def __init__(self, status_code=429, retry_after=None):
        super().__init__("temporary")
        self.status_code = status_code
        self.retry_after = retry_after


class Permanent(Exception):
    status_code = 400


def is_transient(exc):
    return isinstance(exc, Temporary)


def retry_after(exc):
    return getattr(exc, "retry_after", None)


def flaky(*outcomes):
    """A call that raises or returns each outcome in turn, counting calls."""
    calls = []

    def fn():
        outcome = outcomes[len(calls)]
        calls.append(outcome)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    return fn, calls


def run(fn, jitter=0.5):
    """with_retries with a recording sleep and a fixed jitter; -> (result or exception, sleeps)."""
    sleeps = []
    try:
        result = with_retries(fn, is_transient=is_transient, retry_after=retry_after,
                              sleep=sleeps.append, rand=lambda: jitter)
    except Exception as exc:  # noqa: BLE001 - the tests inspect what was raised
        return exc, sleeps
    return result, sleeps


def test_temporary_error_then_success_returns_the_reply():
    fn, calls = flaky(Temporary(), "ok")
    result, sleeps = run(fn)
    assert result == "ok"
    assert len(calls) == 2
    assert sleeps == [1.5]


def test_retries_exhausted_raises_the_original_exception():
    errors = [Temporary() for _ in range(MAX_RETRIES + 1)]
    fn, calls = flaky(*errors)
    result, sleeps = run(fn)
    assert result is errors[-1]
    assert len(calls) == MAX_RETRIES + 1 == 4
    assert len(sleeps) == MAX_RETRIES


def test_permanent_error_is_not_retried():
    error = Permanent()
    fn, calls = flaky(error, "never reached")
    result, sleeps = run(fn)
    assert result is error
    assert len(calls) == 1
    assert sleeps == []


def test_permanent_error_after_a_retry_stops_retrying():
    error = Permanent()
    fn, calls = flaky(Temporary(), error, "never reached")
    result, sleeps = run(fn)
    assert result is error
    assert len(calls) == 2
    assert len(sleeps) == 1


def test_waits_double_with_jitter_added():
    fn, _ = flaky(Temporary(), Temporary(), Temporary(), "ok")
    result, sleeps = run(fn, jitter=0.25)
    assert result == "ok"
    assert sleeps == [1.25, 2.25, 4.25]


def test_small_retry_after_is_respected():
    fn, _ = flaky(Temporary(retry_after=3.0), "ok")
    result, sleeps = run(fn, jitter=0.5)
    assert result == "ok"
    assert sleeps == [3.0]


def test_retry_after_shorter_than_backoff_keeps_the_backoff():
    fn, _ = flaky(Temporary(retry_after=0.1), "ok")
    _, sleeps = run(fn, jitter=0.5)
    assert sleeps == [1.5]


def test_large_retry_after_gives_up_without_waiting():
    error = Temporary(retry_after=60.0)
    fn, calls = flaky(error, "never reached")
    result, sleeps = run(fn)
    assert result is error
    assert len(calls) == 1
    assert sleeps == []


def test_total_wait_never_exceeds_the_cap():
    # 20s fits; the next 20s would bring the total to 40s, so it gives up.
    error = Temporary(retry_after=20.0)
    fn, calls = flaky(Temporary(retry_after=20.0), error, "never reached")
    result, sleeps = run(fn)
    assert result is error
    assert len(calls) == 2
    assert sleeps == [20.0]
    assert sum(sleeps) <= MAX_TOTAL_WAIT == 30.0


def test_retry_after_exactly_filling_the_budget_is_allowed():
    fn, _ = flaky(Temporary(retry_after=MAX_TOTAL_WAIT), "ok")
    result, sleeps = run(fn)
    assert result == "ok"
    assert sleeps == [MAX_TOTAL_WAIT]


def test_non_exception_base_errors_are_never_retried():
    fn, calls = flaky(KeyboardInterrupt(), "never reached")
    with pytest.raises(KeyboardInterrupt):
        with_retries(fn, is_transient=lambda exc: True, retry_after=retry_after,
                     sleep=lambda s: None, rand=lambda: 0.0)
    assert len(calls) == 1


def test_each_retry_is_logged_to_stderr_without_the_prompt(capsys):
    prompt = "SECRET-PROMPT merchant=Acme amount=123.45"
    fn, _ = flaky(Temporary(status_code=529), Temporary(status_code=503), "ok")
    with_retries(fn, is_transient=is_transient, retry_after=retry_after,
                 sleep=lambda s: None, rand=lambda: 0.4)
    out, err = capsys.readouterr()
    assert out == ""
    lines = err.strip().splitlines()
    assert lines == [
        "[call_model] retry 1/3 after Temporary (status 529); waiting 1.4s",
        "[call_model] retry 2/3 after Temporary (status 503); waiting 2.4s",
    ]
    assert prompt not in err
    assert "temporary" not in err  # str(exc) is never logged


def test_giving_up_on_a_large_retry_after_is_logged(capsys):
    fn, _ = flaky(Temporary(retry_after=45.0))
    with pytest.raises(Temporary):
        with_retries(fn, is_transient=is_transient, retry_after=retry_after,
                     sleep=lambda s: None, rand=lambda: 0.0)
    err = capsys.readouterr().err
    assert "giving up" in err
    assert "45.0s" in err


def test_status_is_omitted_when_the_error_has_none(capsys):
    class Timeout(Exception):
        pass
    fn, _ = flaky(Timeout(), "ok")
    with_retries(fn, is_transient=lambda exc: True, retry_after=lambda exc: None,
                 sleep=lambda s: None, rand=lambda: 0.0)
    assert capsys.readouterr().err.strip() == "[call_model] retry 1/3 after Timeout; waiting 1.0s"


def test_defaults_are_real_sleep_and_random():
    import inspect
    import random
    import time
    parameters = inspect.signature(with_retries).parameters
    assert parameters["sleep"].default is time.sleep
    assert parameters["rand"].default is random.random
