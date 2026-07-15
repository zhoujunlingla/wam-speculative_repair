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
        '{"source":"draft_flash","elapsed_sec":0.4,"accepted_prefix":16,'
        '"repair_eligible":true,"repair_attempted":true,'
        '"repair_executed":true,"repair_kind":"zero_prefix_endpoint",'
        '"primary_verify_forwards":2,"repair_construction_forwards":1,'
        '"repair_holdout_forwards":2,"repair_holdout_evaluated":true,'
        '"repair_action_only_forwards":3}\n'
        '{"source":"teacher_full","elapsed_sec":1.2,"accepted_prefix":32}\n'
    )

    summary = source_summary(path)

    assert summary["counts"] == {"draft_flash": 1, "teacher_full": 1}
    assert summary["teacher_action_rate"] == 0.5
    assert summary["model_time"]["executed_actions"] == 48
    assert summary["model_time"]["action_path_hz"] == pytest.approx(30.0)
    assert summary["repair"] == {
        "eligible": 1,
        "attempted": 1,
        "executed": 1,
        "primary_verify_forwards": 2,
        "construction_forwards": 1,
        "holdout_forwards": 2,
        "holdout_evaluated": 1,
        "action_only_forwards": 3,
        "kinds": {"zero_prefix_endpoint": 1},
        "episodes_executed": 0,
        "episodes_success": 0,
        "episode_success_rate": None,
        "episode_kinds": {},
        "episode_log_count": 0,
        "episode_outcome_count": 0,
        "episode_alignment_valid": True,
    }


def test_source_summary_reports_repaired_episode_success(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"source":"reset"}\n'
        '{"source":"draft_flash","repair_executed":true,'
        '"repair_kind":"zero_prefix_flow_rk2"}\n'
        '{"source":"reset"}\n'
        '{"source":"teacher_full"}\n'
    )

    repair = source_summary(path, {0: True, 1: False})["repair"]

    assert repair["episodes_executed"] == 1
    assert repair["episodes_success"] == 1
    assert repair["episode_success_rate"] == 1.0
    assert repair["episode_kinds"] == {
        "zero_prefix_flow_rk2": {"episodes": 1, "success": 1}
    }
    assert repair["episode_alignment_valid"] is True


def test_source_summary_rejects_misaligned_episode_outcomes(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"source":"reset"}\n{"source":"teacher_full"}\n')

    repair = source_summary(path, {1: True})["repair"]

    assert repair["episode_log_count"] == 1
    assert repair["episode_outcome_count"] == 1
    assert repair["episode_alignment_valid"] is False
    assert repair["episodes_executed"] == 0


def test_source_summary_counts_flowguard_shadow_counterfactuals(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"source":"replan","flowguard_shadow_enabled":true,'
        '"flowguard_class":"near_miss","flowguard_shadow_eligible":true,'
        '"flowguard_shadow_rescue":true,"flowguard_original_holdout_prefix":0,'
        '"flowguard_repaired_holdout_prefix":16,'
        '"flowguard_shadow_action_only_forwards":4}\n'
        '{"source":"draft_flash","flowguard_shadow_enabled":true,'
        '"flowguard_class":"pass","flowguard_k1_continuous_would_accept":true,'
        '"flowguard_k1_false_accept":true}\n'
    )

    flowguard = source_summary(path)["flowguard_shadow"]

    assert flowguard["classes"] == {"near_miss": 1, "pass": 1}
    assert flowguard["eligible"] == 1
    assert flowguard["rescued"] == 1
    assert flowguard["rescue_rate"] == 1.0
    assert flowguard["prefix_improved"] == 1
    assert flowguard["prefix_worsened"] == 0
    assert flowguard["action_only_forwards"] == 4
    assert flowguard["projected_full_teacher_rollouts_avoided"] == 1
    assert flowguard["k1_continuous_would_accept"] == 1
    assert flowguard["k1_false_accept"] == 1
