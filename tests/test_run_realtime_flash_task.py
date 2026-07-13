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
            "executed_action_steps": 16,
            "model_timing_ms": {
                "draft_generation.video_dit": 100.0,
                "teacher_verify_primary.teacher_action_verify": 20.0,
            },
            "model_forward_counts": {
                "draft_generation.video_dit": 2,
                "teacher_verify_primary.teacher_action_verify": 2,
            },
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
        {
            "source": "teacher_full",
            "elapsed_sec": 0.9,
            "executed_action_steps": 32,
            "model_timing_ms": {
                "teacher_generation.video_dit": 120.0,
            },
            "model_forward_counts": {
                "teacher_generation.video_dit": 3,
            },
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    summary = source_summary(path)

    assert summary["verify_k_counts"] == {"2": 2}
    assert summary["action_steps_by_source"] == {
        "draft_flash": 16,
        "teacher_full": 32,
    }
    assert summary["teacher_action_step_rate"] == 2 / 3
    assert summary["teacher_verify_forwards_per_100_actions"] == 100.0 / 24.0
    assert summary["teacher_model_ms_per_100_actions"] == 875.0 / 3.0
    assert summary["motion_selective"] == {
        "high_proposals": 2,
        "high_accepts": 1,
        "high_rejects": 1,
        "high_prefix_caps": 1,
        "high_accept_rate": 0.5,
    }
    assert summary["model_only"] == {
        "total_ms": 240.0,
        "executed_action_steps": 48,
        "action_hz": 200.0,
        "components_ms": {
            "draft_generation.video_dit": 100.0,
            "teacher_verify_primary.teacher_action_verify": 20.0,
            "teacher_generation.video_dit": 120.0,
        },
        "forward_counts": {
            "draft_generation.video_dit": 2,
            "teacher_verify_primary.teacher_action_verify": 2,
            "teacher_generation.video_dit": 3,
        },
    }
