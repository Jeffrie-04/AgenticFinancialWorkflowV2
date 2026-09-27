"""
tests/test_network_guard.py — proves the autouse guard in conftest.py: no
test can reach a provider. Credentials are removed, .env is not loaded, and
any outgoing connection fails loudly with NetworkBlocked, which is not an
Exception, so neither the SDKs nor call_model's retries can swallow it.
"""
import os
import socket

import pytest
from conftest import NetworkBlocked

import bedrock_client

CREDENTIALS = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL",
               "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"]


def test_provider_credentials_are_removed():
    for name in CREDENTIALS:
        assert name not in os.environ, name
    # ~/.aws is not read either
    assert not os.path.exists(os.environ["AWS_SHARED_CREDENTIALS_FILE"])
    assert not os.path.exists(os.environ["AWS_CONFIG_FILE"])


def test_dotenv_is_not_loaded(tmp_path):
    from dotenv import load_dotenv
    env = tmp_path / ".env"
    env.write_text("NETWORK_GUARD_SENTINEL=loaded\n")
    assert load_dotenv(env) is False
    assert "NETWORK_GUARD_SENTINEL" not in os.environ


def test_outgoing_connections_fail_loudly():
    with pytest.raises(NetworkBlocked, match="93.184.216.34"):
        socket.create_connection(("93.184.216.34", 443), timeout=1)
    with pytest.raises(NetworkBlocked, match="api.anthropic.com"):
        socket.getaddrinfo("api.anthropic.com", 443)


def test_loopback_is_still_allowed():
    """In-process servers (e.g. streamlit's AppTest) keep working."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=1):
            pass


@pytest.fixture
def dummy_credentials(monkeypatch):
    """Fake keys so each SDK gets past its own auth check and actually tries to connect."""
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")


@pytest.mark.parametrize("provider", ["openai", "anthropic_direct", "claude"])
def test_unstubbed_provider_call_is_blocked_not_sent(monkeypatch, dummy_credentials, provider):
    sleeps = []
    monkeypatch.setattr(bedrock_client, "MODEL_PROVIDER", provider)
    monkeypatch.setattr(bedrock_client, "_sleep", sleeps.append)

    with pytest.raises(NetworkBlocked):
        bedrock_client.call_model("the prompt")
    assert sleeps == []  # not mistaken for a temporary error and retried
