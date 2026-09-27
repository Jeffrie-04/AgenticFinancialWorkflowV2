"""
tests/test_bedrock_retry.py — which provider errors bedrock_client treats as
temporary, and the Retry-After each provider sent. Exceptions are real SDK
instances; no network. httpx2 is the SDKs' own transport, not a new dependency.
"""
import anthropic
import httpx2
import openai
import pytest
from botocore import exceptions as boto

from bedrock_client import _is_transient, _retry_after, _status_of

REQUEST = httpx2.Request("POST", "https://example.test/v1")


def sdk_status_error(sdk, status, headers=None):
    """The exception the SDK itself raises for this status (e.g. RateLimitError for 429)."""
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
