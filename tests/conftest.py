import ipaddress
import os
import socket
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set before any test module imports bedrock_client, whose load_dotenv() would
# otherwise read the developer's real keys from .env (python-dotenv >= 1.2).
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

CREDENTIALS = ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL")
NO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "no-such-aws-file")


class NetworkBlocked(BaseException):
    """A test tried to open an outgoing connection. Deliberately not an
    Exception: the SDKs wrap Exceptions as connection errors and call_model
    would retry them, which would hide the attempt instead of failing loudly."""


def _is_local(host):
    if host is None or host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(str(host).split("%", 1)[0]).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def no_network_no_keys(monkeypatch):
    """Every test: no provider credentials, no .env, no outgoing connections.
    Loopback and Unix sockets stay allowed (streamlit's AppTest runs in-process)."""
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    for name in CREDENTIALS:
        monkeypatch.delenv(name, raising=False)
    for name in [n for n in os.environ if n.startswith("AWS_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", NO_FILE)  # don't read ~/.aws
    monkeypatch.setenv("AWS_CONFIG_FILE", NO_FILE)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")

    real_getaddrinfo = socket.getaddrinfo
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def getaddrinfo(host, *args, **kwargs):
        if not _is_local(host):
            raise NetworkBlocked(f"test tried to resolve {host!r}: the network is blocked in tests")
        return real_getaddrinfo(host, *args, **kwargs)

    def check(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(address[0]):
            raise NetworkBlocked(f"test tried to connect to {address!r}: the network is blocked in tests")

    def connect(sock, address):
        check(sock, address)
        return real_connect(sock, address)

    def connect_ex(sock, address):
        check(sock, address)
        return real_connect_ex(sock, address)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
