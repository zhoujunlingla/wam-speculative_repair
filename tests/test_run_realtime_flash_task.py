import json
import socket

import pytest

from scripts.run_realtime_flash_task import runtime_env, source_summary, wait_for_server


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


def test_source_summary_distinguishes_k1_and_k2_world_flow_evidence(tmp_path):
    trace = tmp_path / "specverify.jsonl"
    rows = [
        {
            "source": "draft_flash",
            "requested_verify_k": 2,
            "effective_verify_k": 1,
            "primary_verify_forwards": 1,
            "world_flow_evidence": {
                "available": False,
                "reason": "requires_completed_k2",
            },
            "action_dynamics_stats": {"jerk_rms": 0.2},
            "video_motion_stats": {"global_mean": 0.4},
        },
        {
            "source": "draft_flash",
            "requested_verify_k": 2,
            "effective_verify_k": 2,
            "primary_verify_forwards": 2,
            "world_flow_evidence": {
                "available": True,
                "endpoint_midpoint_distance_mean": 0.1,
                "cross_tau_half_gap_mean": 0.02,
                "correction_cosine_mean_valid": 0.9,
            },
        },
    ]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))

    summary = source_summary(trace)

    assert summary["world_flow"]["verify_calls"] == 2
    assert summary["world_flow"]["unavailable_k1"] == 1
    assert summary["world_flow"]["available_k2"] == 1
    assert summary["world_flow"]["missing"] == 0
    assert summary["world_flow"]["cross_tau_half_gap_mean"]["mean"] == 0.02
    assert summary["draft_evidence"]["jerk_rms"]["mean"] == 0.2


def test_source_summary_marks_k2_telemetry_error_missing(tmp_path):
    trace = tmp_path / "specverify.jsonl"
    trace.write_text(json.dumps({
        "source": "draft_flash",
        "requested_verify_k": 2,
        "effective_verify_k": 2,
        "primary_verify_forwards": 2,
        "world_flow_evidence": {
            "available": False,
            "reason": "telemetry_error:RuntimeError",
        },
    }) + "\n")

    summary = source_summary(trace)

    assert summary["world_flow"]["available_k2"] == 0
    assert summary["world_flow"]["unavailable_k1"] == 0
    assert summary["world_flow"]["missing"] == 1


def test_source_summary_does_not_count_nonfinite_evidence(tmp_path):
    trace = tmp_path / "specverify.jsonl"
    trace.write_text(json.dumps({
        "source": "draft_flash",
        "action_dynamics_stats": {"jerk_rms": float("nan")},
        "video_motion_stats": {"global_mean": float("inf")},
    }) + "\n")

    summary = source_summary(trace)

    assert summary["draft_evidence"]["action_dynamics_calls"] == 1
    assert summary["draft_evidence"]["video_motion_calls"] == 1
    assert summary["draft_evidence"]["jerk_rms"]["count"] == 0
    assert summary["draft_evidence"]["video_global_mean"]["count"] == 0


def test_source_summary_rejects_available_evidence_from_k1(tmp_path):
    trace = tmp_path / "specverify.jsonl"
    trace.write_text(json.dumps({
        "source": "draft_flash",
        "requested_verify_k": 2,
        "effective_verify_k": 1,
        "primary_verify_forwards": 1,
        "world_flow_evidence": {
            "available": True,
            "endpoint_midpoint_distance_mean": 0.1,
            "cross_tau_half_gap_mean": 0.02,
        },
    }) + "\n")

    summary = source_summary(trace)

    assert summary["world_flow"]["available_k2"] == 0
    assert summary["world_flow"]["unavailable_k1"] == 0
    assert summary["world_flow"]["missing"] == 1
