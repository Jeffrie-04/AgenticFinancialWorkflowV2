"""
tests/test_bedrock_retry.py — which provider errors bedrock_client treats as
temporary, and the Retry-After each provider sent. Exceptions are real SDK
instances; no network. httpx2 is the SDKs' own transport, not a new dependency.
"""
import os

import anthropic
import httpx2
import openai
import pytest
from botocore import exceptions as boto

import bedrock_client
import run
from bedrock_client import _is_transient, _retry_after, _status_of

REQUEST = httpx2.Request("POST", "https://example.test/v1")
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def sdk_status_error(sdk, status, headers=None):
    """The exception the SDK itself raises for this status (e.g. RateLimitError for 429).

    Deliberately uses the SDK's private _make_status_error rather than building
    exception classes by hand: these tests then prove the SDK's real status ->
    exception mapping (e.g. 529 -> anthropic.OverloadedError), which is what
    call_model will actually see. Both SDK versions are pinned in
    requirements.txt; if an upgrade breaks this helper, that is the signal the
    mapping changed and the classification must be re-checked."""
    response = httpx2.Response(status, request=REQUEST, headers=headers or {})
    client = sdk.Anthropic(api_key="test") if sdk is anthropic else sdk.OpenAI(api_key="test")
    return client._make_status_error("error", body=None, response=response)


def client_error(code, status, headers=None):
    return boto.ClientError({"Error": {"Code": code, "Message": "m"},
                             "ResponseMetadata": {"HTTPStatusCode": status, "HTTPHeaders": headers or {}}},
                            "InvokeModel")


# ------------------------------------------------ Anthropic direct and OpenAI


@pytest.mark.parametrize("sdk", [anthropic, openai], ids=["anthropic", "openai"])
@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 529])
def test_sdk_temporary_statuses(sdk, status):
    exc = sdk_status_error(sdk, status)
    assert isinstance(exc, sdk.APIStatusError)
    assert _is_transient(exc)
    assert _status_of(exc) == status


@pytest.mark.parametrize("sdk", [anthropic, openai], ids=["anthropic", "openai"])
@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 413, 422])
def test_sdk_permanent_statuses(sdk, status):
    assert not _is_transient(sdk_status_error(sdk, status))


def test_sdk_named_classes_for_the_temporary_statuses():
    assert isinstance(sdk_status_error(anthropic, 429), anthropic.RateLimitError)
    assert isinstance(sdk_status_error(anthropic, 529), anthropic.OverloadedError)
    assert isinstance(sdk_status_error(anthropic, 500), anthropic.InternalServerError)
    assert isinstance(sdk_status_error(openai, 429), openai.RateLimitError)
    assert isinstance(sdk_status_error(openai, 503), openai.InternalServerError)


@pytest.mark.parametrize("sdk", [anthropic, openai], ids=["anthropic", "openai"])
def test_sdk_timeouts_and_connection_errors_are_temporary(sdk):
    for exc in (sdk.APITimeoutError(request=REQUEST), sdk.APIConnectionError(request=REQUEST)):
        assert _is_transient(exc)
        assert _status_of(exc) is None


@pytest.mark.parametrize("sdk", [anthropic, openai], ids=["anthropic", "openai"])
def test_sdk_retry_after(sdk):
    assert _retry_after(sdk_status_error(sdk, 429, {"retry-after": "7"})) == 7.0
    assert _retry_after(sdk_status_error(sdk, 429, {"Retry-After": "1.5"})) == 1.5
    # retry-after-ms is more precise and wins when both are sent
    assert _retry_after(sdk_status_error(sdk, 429, {"retry-after-ms": "2500", "retry-after": "9"})) == 2.5
    assert _retry_after(sdk_status_error(sdk, 429)) is None
    assert _retry_after(sdk.APITimeoutError(request=REQUEST)) is None


@pytest.mark.parametrize("value", ["Wed, 21 Oct 2026 07:28:00 GMT", "soon", "", "-3", "nan", "inf"])
def test_unparseable_retry_after_is_ignored(value):
    assert _retry_after(sdk_status_error(anthropic, 429, {"retry-after": value})) is None


# ------------------------------------------------------------- Bedrock native


@pytest.mark.parametrize("code,status", [
    ("ThrottlingException", 429),
    ("ServiceUnavailableException", 503),
    ("InternalServerException", 500),
    ("ModelNotReadyException", 429),
    ("ModelTimeoutException", 408),
    ("SomethingNew", 502),        # unknown code, 5xx status
])
def test_bedrock_temporary_client_errors(code, status):
    exc = client_error(code, status)
    assert _is_transient(exc)
    assert _status_of(exc) == status


@pytest.mark.parametrize("code,status", [
    ("ValidationException", 400),
    ("UnrecognizedClientException", 403),
    ("AccessDeniedException", 403),
    ("ResourceNotFoundException", 404),
    ("ModelErrorException", 424),
    ("ServiceQuotaExceededException", 400),
])
def test_bedrock_permanent_client_errors(code, status):
    assert not _is_transient(client_error(code, status))


def test_bedrock_throttling_code_without_metadata_is_temporary():
    exc = boto.ClientError({"Error": {"Code": "ThrottlingException"}}, "InvokeModel")
    assert _is_transient(exc)
    assert _status_of(exc) is None
    assert _retry_after(exc) is None


@pytest.mark.parametrize("exc", [
    boto.ReadTimeoutError(endpoint_url="u"),
    boto.ConnectTimeoutError(endpoint_url="u"),
    boto.EndpointConnectionError(endpoint_url="u"),
    boto.ConnectionClosedError(endpoint_url="u"),
], ids=lambda e: type(e).__name__)
def test_bedrock_timeouts_and_connection_errors_are_temporary(exc):
    assert _is_transient(exc)
    assert _status_of(exc) is None


def test_bedrock_retry_after():
    assert _retry_after(client_error("ThrottlingException", 429, {"retry-after": "4"})) == 4.0
    assert _retry_after(client_error("ThrottlingException", 429)) is None


def test_bedrock_other_botocore_errors_are_permanent():
    assert not _is_transient(boto.NoCredentialsError())
    assert not _is_transient(boto.ParamValidationError(report="bad"))


# ------------------------------------------------------------ everything else


@pytest.mark.parametrize("exc", [
    ValueError("x"), KeyError("content"), TypeError("x"), RuntimeError("x"),
    anthropic.AuthenticationError("x", response=httpx2.Response(401, request=REQUEST), body=None),
    openai.BadRequestError("x", response=httpx2.Response(400, request=REQUEST), body=None),
], ids=lambda e: type(e).__name__)
def test_non_provider_and_permanent_errors_are_not_retried(exc):
    assert not _is_transient(exc)


def test_classification_does_not_read_the_message():
    """A permanent error whose message mentions rate limits is still permanent."""
    exc = openai.BadRequestError("rate limit exceeded; overloaded; timeout",
                                 response=httpx2.Response(400, request=REQUEST), body=None)
    assert not _is_transient(exc)


# ------------------------------------ clients: no SDK retries, 60s timeout


@pytest.fixture
def dummy_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")


def test_openai_client_has_no_retries_and_a_60s_timeout(dummy_keys):
    client = bedrock_client._openai_client()
    assert client.max_retries == 0
    assert client.timeout == 60 == bedrock_client.MODEL_TIMEOUT


def test_anthropic_client_has_no_retries_and_a_60s_timeout(dummy_keys):
    client = bedrock_client._anthropic_client()
    assert client.max_retries == 0
    assert client.timeout == 60


def test_bedrock_client_makes_one_attempt_with_a_60s_timeout(dummy_keys):
    config = bedrock_client._bedrock_client().meta.config
    assert config.retries == {"total_max_attempts": 1, "mode": "standard"}
    assert config.read_timeout == 60
    assert config.connect_timeout == 60


# ------------------------------------------------- call_model uses the policy


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(bedrock_client, "_sleep", recorded.append)
    monkeypatch.setattr(bedrock_client, "_rand", lambda: 0.5)
    return recorded


def replies(monkeypatch, *outcomes, provider="openai"):
    """Make the provider call raise or return each outcome in turn; -> prompts sent."""
    prompts = []
    target = {"openai": "_call_openai", "claude": "_call_claude",
              "anthropic_direct": "_call_anthropic_direct"}[provider]

    def fake(prompt):
        outcome = outcomes[len(prompts)]
        prompts.append(prompt)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    monkeypatch.setattr(bedrock_client, "MODEL_PROVIDER", provider)
    monkeypatch.setattr(bedrock_client, target, fake)
    return prompts


@pytest.mark.parametrize("provider,error", [
    ("openai", lambda: sdk_status_error(openai, 429)),
    ("anthropic_direct", lambda: sdk_status_error(anthropic, 529)),
    ("claude", lambda: boto.ReadTimeoutError(endpoint_url="u")),
])
def test_call_model_retries_a_temporary_error(monkeypatch, sleeps, provider, error):
    prompts = replies(monkeypatch, error(), "ok", provider=provider)
    assert bedrock_client.call_model("the prompt") == "ok"
    assert prompts == ["the prompt", "the prompt"]
    assert sleeps == [1.5]


def test_call_model_does_not_retry_a_permanent_error(monkeypatch, sleeps):
    error = sdk_status_error(openai, 401)
    prompts = replies(monkeypatch, error, "never reached")
    with pytest.raises(openai.AuthenticationError) as raised:
        bedrock_client.call_model("the prompt")
    assert raised.value is error
    assert len(prompts) == 1
    assert sleeps == []


def test_call_model_raises_after_three_retries(monkeypatch, sleeps):
    errors = [sdk_status_error(anthropic, 529) for _ in range(4)]
    prompts = replies(monkeypatch, *errors, provider="anthropic_direct")
    with pytest.raises(anthropic.OverloadedError) as raised:
        bedrock_client.call_model("the prompt")
    assert raised.value is errors[-1]
    assert len(prompts) == 4
    assert sleeps == [1.5, 2.5, 4.5]


def test_call_model_uses_retry_after_and_respects_the_budget(monkeypatch, sleeps):
    prompts = replies(monkeypatch, sdk_status_error(openai, 429, {"retry-after": "45"}), "never reached")
    with pytest.raises(openai.RateLimitError):
        bedrock_client.call_model("the prompt")
    assert len(prompts) == 1
    assert sleeps == []


def test_call_model_logs_retries_without_the_prompt(monkeypatch, sleeps, capsys):
    replies(monkeypatch, sdk_status_error(openai, 429), "ok")
    bedrock_client.call_model("SECRET-PROMPT merchant=Acme")
    err = capsys.readouterr().err
    assert "retry 1/3 after RateLimitError (status 429); waiting 1.5s" in err
    assert "SECRET-PROMPT" not in err


def test_an_outage_still_stops_run_py_with_exit_1(tmp_path, monkeypatch, sleeps, capsys):
    """Retries exhausted in the first model phase (the plan) -> run.py returns 1."""
    import shutil
    shutil.copy(os.path.join(FIXTURES, "pipeline", "codex_finding1.csv"), tmp_path / "transactions.csv")
    prompts = replies(monkeypatch, *[sdk_status_error(openai, 503) for _ in range(4)])

    assert run.main(business_dir=str(tmp_path)) == 1

    assert len(prompts) == 4
    assert len(sleeps) == 3
    out = capsys.readouterr().out
    assert "FAILED at Phase 2 - Plan: InternalServerError" in out
    assert not (tmp_path / "outputs" / "plan.json").exists()
