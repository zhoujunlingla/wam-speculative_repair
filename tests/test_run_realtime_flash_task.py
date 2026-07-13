import json
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


def test_source_summary_aggregates_motion_selective_metrics(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = [
        {
            "source": "draft_flash",
            "elapsed_sec": 0.4,
            "active_tau_timesteps": [50.0, 100.0],
            "high_video_motion": True,
            "motion_prefix_cap_applied": True,
        },
        {
            "source": "replan",
            "elapsed_sec": 0.3,
            "active_tau_timesteps": [50.0, 100.0],
            "high_video_motion": True,
            "motion_prefix_cap_applied": False,
        },
        {"source": "teacher_full", "elapsed_sec": 0.9},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    summary = source_summary(path)

    assert summary["verify_k_counts"] == {"2": 2}
    assert summary["motion_selective"] == {
        "high_proposals": 2,
        "high_accepts": 1,
        "high_rejects": 1,
        "high_prefix_caps": 1,
        "high_accept_rate": 0.5,
    }
