import socket

import pytest

from scripts.run_realtime_flash_task import runtime_env, wait_for_server


class _Process:
    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode


def test_wait_for_server_uses_tcp_readiness():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    try:
        wait_for_server(_Process(), listener.getsockname()[1], timeout=1)
    finally:
        listener.close()


def test_wait_for_server_fails_when_process_exits():
    with pytest.raises(RuntimeError, match="server exited"):
        wait_for_server(_Process(returncode=3), 1, timeout=1)


def test_deterministic_profile_is_server_audit_only(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    ordinary = runtime_env(2, server=True)
    audited = runtime_env(2, server=True, deterministic_audit=True)
    client = runtime_env(2, server=False, deterministic_audit=True)

    assert "CUBLAS_WORKSPACE_CONFIG" not in ordinary
    assert audited["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert "CUBLAS_WORKSPACE_CONFIG" not in client
