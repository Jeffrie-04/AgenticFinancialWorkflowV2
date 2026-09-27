import json
import math
import os
import random
import ssl
import time
from email.utils import parsedate_to_datetime

import anthropic
import boto3
import openai
from anthropic import Anthropic
from botocore import exceptions as boto_exceptions
from botocore.config import Config
from openai import OpenAI

from afw.retry import with_retries

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency for local development
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

# Which backend call_model() actually uses. Default is "openai" (GPT-5.6 Terra
# via bedrock-mantle) since that's what currently has quota; set
# MODEL_PROVIDER=claude in the environment to fall back to Bedrock's native
# Claude Haiku invoke_model path once its quota clears — no code changes needed.
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "openai")

OPENAI_MODEL_ID = "us.openai.gpt-5.6-terra"
CLAUDE_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
ANTHROPIC_MODEL_ID = "claude-haiku-4-5"  # direct API alias — no date suffix

# Seconds per attempt, for all three clients. A timeout is a temporary error,
# so call_model retries it under afw/retry.py.
MODEL_TIMEOUT = 60

# The SDKs' own retries are off (one attempt each): call_model's retry policy
# is the only one, so retries never multiply.
CLAUDE_CONFIG = Config(read_timeout=MODEL_TIMEOUT, connect_timeout=MODEL_TIMEOUT,
                       retries={'total_max_attempts': 1, 'mode': 'standard'})

# Injected into afw.retry.with_retries; tests replace them so nothing sleeps.
_sleep = time.sleep
_rand = random.random


# ------------------------------------------- temporary vs. permanent errors
# Used by the retry policy in afw/retry.py. Only temporary errors are retried:
# rate limits (429), overloaded (529), any 5xx, request timeouts (408),
# timeouts, connection errors and a truncated response body. Everything else
# (400, 401, 403, 404, 409, 413, 422, a failed certificate check, parse
# errors, our own exceptions) fails at once. Decided by type, status code and
# the exception's cause chain, never by the error message.

TRANSIENT_STATUSES = frozenset({408, 429})  # plus every status >= 500, which includes 529

SDK_STATUS_ERRORS = (anthropic.APIStatusError, openai.APIStatusError)
# Timeouts are subclasses: anthropic/openai.APITimeoutError.
SDK_CONNECTION_ERRORS = (anthropic.APIConnectionError, openai.APIConnectionError)
# ConnectionError covers ConnectTimeoutError and EndpointConnectionError;
# HTTPClientError covers ReadTimeoutError and ConnectionClosedError;
# IncompleteReadError is a response body cut short (raised by body.read()).
BOTO_CONNECTION_ERRORS = (boto_exceptions.ConnectionError, boto_exceptions.HTTPClientError,
                          boto_exceptions.IncompleteReadError)
BEDROCK_TRANSIENT_CODES = frozenset({
    "ThrottlingException", "ServiceUnavailableException", "InternalServerException",
    "ModelNotReadyException", "ModelTimeoutException",
})


def _status_of(exc):
    """HTTP status of a provider error, or None (timeouts, connection errors, others)."""
    if isinstance(exc, SDK_STATUS_ERRORS):
        return exc.status_code
    if isinstance(exc, boto_exceptions.ClientError):
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return status if isinstance(status, int) else None
    return None


def _causes(exc):
    """exc and every exception it wraps: __cause__, __context__, botocore's
    error= keyword, and exceptions held in args (urllib3's SSLError(e))."""
    seen, pending = set(), [exc]
    while pending:
        current = pending.pop()
        if not isinstance(current, BaseException) or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending += [current.__cause__, current.__context__,
                    getattr(current, "kwargs", {}).get("error"), *current.args]


def _is_certificate_failure(exc):
    """A failed TLS certificate check (wrong CA, expired, bad hostname) anywhere
    in the chain. Retrying can't fix it, so it is permanent even though the SDK
    reports it as a connection error."""
    return any(isinstance(cause, ssl.SSLCertVerificationError) for cause in _causes(exc))


def _is_transient(exc):
    if isinstance(exc, SDK_CONNECTION_ERRORS + BOTO_CONNECTION_ERRORS):
        return not _is_certificate_failure(exc)
    if isinstance(exc, boto_exceptions.ClientError) and \
            exc.response.get("Error", {}).get("Code") in BEDROCK_TRANSIENT_CODES:
        return True
    if isinstance(exc, SDK_STATUS_ERRORS + (boto_exceptions.ClientError,)):
        status = _status_of(exc)
        return status is not None and (status in TRANSIENT_STATUSES or status >= 500)
    return False


_now = time.time  # replaced in tests


def _seconds(value, scale=1.0):
    """A header value as a finite, non-negative number of seconds, else None."""
    try:
        seconds = float(value) / scale
    except (TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _seconds_until(value):
    """An HTTP-date (RFC 9110, e.g. "Wed, 21 Oct 2026 07:28:00 GMT") as seconds
    from now; 0 if it is already past; None if it isn't a valid date."""
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:  # RFC 9110 dates are GMT
        return None
    return max(0.0, when.timestamp() - _now())


def _retry_after(exc):
    """Seconds the provider asked us to wait, or None: retry-after-ms if it is
    valid, else retry-after as seconds or an HTTP-date. The retry budget in
    afw/retry.py applies to whatever this returns."""
    if isinstance(exc, SDK_STATUS_ERRORS):
        headers = exc.response.headers  # case-insensitive
    elif isinstance(exc, boto_exceptions.ClientError):
        raw = exc.response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
        headers = {str(k).lower(): v for k, v in raw.items()}
    else:
        return None
    milliseconds = _seconds(headers.get("retry-after-ms"), scale=1000.0)
    if milliseconds is not None:
        return milliseconds
    value = headers.get("retry-after")
    if value is None:
        return None
    seconds = _seconds(value)
    return seconds if seconds is not None else _seconds_until(value)


# ------------------------------------------------------------------ clients


def _openai_client():
    # Reads OPENAI_API_KEY / OPENAI_BASE_URL from the environment.
    return OpenAI(max_retries=0, timeout=MODEL_TIMEOUT)


def _anthropic_client():
    # Reads ANTHROPIC_API_KEY from the environment.
    return Anthropic(max_retries=0, timeout=MODEL_TIMEOUT)


def _bedrock_client():
    return boto3.client('bedrock-runtime', region_name='us-east-1', config=CLAUDE_CONFIG)


def _call_openai(prompt):
    client = _openai_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL_ID,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _call_claude(prompt):
    bedrock = _bedrock_client()
    response = bedrock.invoke_model(
        modelId=CLAUDE_MODEL_ID,
        contentType='application/json',
        accept='application/json',
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8000,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }),
    )
    response_body = json.loads(response['body'].read())
    return response_body['content'][0]['text']


def _call_anthropic_direct(prompt):
    client = _anthropic_client()
    # NOTE: `temperature` is not a valid Messages.create() parameter in the
    # installed anthropic SDK (1.6.0) — it's been removed from the Messages
    # API entirely, not just rejected for certain models. Confirmed via
    # direct signature introspection, not assumed.
    response = client.messages.create(
        model=ANTHROPIC_MODEL_ID,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _dispatch(prompt):
    if MODEL_PROVIDER == "claude":
        return _call_claude(prompt)
    if MODEL_PROVIDER == "anthropic_direct":
        return _call_anthropic_direct(prompt)
    return _call_openai(prompt)


def call_model(prompt):
    """The model's reply to prompt. Temporary errors are retried (afw/retry.py);
    a permanent error, or the last temporary one, is raised unchanged."""
    return with_retries(lambda: _dispatch(prompt), is_transient=_is_transient,
                        retry_after=_retry_after, status_of=_status_of,
                        sleep=_sleep, rand=_rand)
