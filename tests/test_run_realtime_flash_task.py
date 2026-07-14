import socket

import pytest

from scripts.run_realtime_flash_task import source_summary, wait_for_server


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


def test_source_summary_counts_repair_telemetry(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"source":"draft_flash","elapsed_sec":0.4,'
        '"repair_eligible":true,"repair_attempted":true,'
        '"repair_executed":true,"repair_kind":"zero_prefix_endpoint",'
        '"repair_action_only_forwards":2}\n'
        '{"source":"teacher_full","elapsed_sec":1.2}\n'
    )

    summary = source_summary(path)

    assert summary["counts"] == {"draft_flash": 1, "teacher_full": 1}
    assert summary["teacher_action_rate"] == 0.5
    assert summary["repair"] == {
        "eligible": 1,
        "attempted": 1,
        "executed": 1,
        "action_only_forwards": 2,
        "kinds": {"zero_prefix_endpoint": 1},
    }
