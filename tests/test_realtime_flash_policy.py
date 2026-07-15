from collections import deque
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.robotwin.realtime_flash_policy import (
    RealtimeFlashPolicy,
    bounded_endpoint_repair,
    classify_flowguard_probe,
    first_gripper_switch,
    flowguard_near_miss_repair,
    gripper_phase_snap_repair,
    infer_with_replan,
    longest_safe_prefix,
    normalized_l2_step_distances,
)


def _action(switch_step=None, value=0.0):
    action = np.full((16, 2, 16), value, dtype=np.float32)
    if switch_step is not None:
        for step in range(switch_step, 32):
            frame, offset = divmod(step, 16)
            action[7, frame, offset] = 1.0
    return action


class _FakeModel:
    def __init__(
        self,
        role,
        events,
        action=None,
        verify_results=(),
        delayed_errors=(),
        video_motion_global=0.1,
    ):
        self.role = role
        self.events = events
        self.action = _action(value=0.25) if action is None else action
        self.action_latent = np.zeros((1, 30, 2, 16, 1), dtype=np.float32)
        self.verify_results = deque(verify_results)
        self.delayed_errors = deque(delayed_errors)
        self.video_motion_global = float(video_motion_global)
        self.calls = []
        self.cache = []
        self.frame_st_id = 0

    def infer(self, request):
        if request.get("reset", False):
            kind = "reset"
        elif request.get("compute_kv_cache", False):
            kind = "cache"
        elif request.get("verify_action", False):
            kind = "verify"
        elif request.get("prime_only", False):
            kind = "prime"
        else:
            kind = "action"

        call = {
            "kind": kind,
            "request": request,
            "cache_before": tuple(self.cache),
            "frame_before": self.frame_st_id,
        }
        self.calls.append(call)
        self.events.append((self.role, kind, request.get("tag")))

        if kind == "reset":
            self.cache.clear()
            self.frame_st_id = 0
            return {}
        if kind == "cache":
            self.cache.append(request["tag"])
            self.frame_st_id += int(np.asarray(request["state"]).shape[1])
            if request.get("compare_video_prediction", False):
                latent_nrmse = (
                    self.delayed_errors.popleft() if self.delayed_errors else 0.25
                )
                return {"delayed_video_error": {"latent_nrmse": latent_nrmse}}
            return {}
        if kind == "verify":
            result = self.verify_results.popleft() if self.verify_results else 32
            return dict(result) if isinstance(result, dict) else {"accepted_prefix": result}
        response = {"action": self.action.copy()}
        if self.role == "draft":
            response["action_latent"] = self.action_latent.copy()
            if request.get("return_video_motion_stats", False):
                response["video_motion_stats"] = {
                    "global_mean": self.video_motion_global,
                    "median": 0.05,
                    "top_mean": 0.2,
                    "top_relative": 4.0,
                    "top_concentration": 0.5,
                }
        return response


def _action_request():
    return {
        "obs": {"observation.state": np.zeros(16, dtype=np.float32)},
        "prompt": "test task",
    }


def _cache_request(tag, action):
    return {
        "compute_kv_cache": True,
        "obs": [{"tag": tag}],
        "state": np.asarray(action),
        "tag": tag,
    }


def _anchored_policy(*, pf_interval=2, draft_action=None, verify_results=()):
    events = []
    draft = _FakeModel("draft", events, action=draft_action)
    teacher = _FakeModel("teacher", events, verify_results=verify_results)
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=pf_interval,
        rng=np.random.default_rng(7),
    )
    policy.infer({"reset": True, "prompt": "test task"})
    first = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", first["action"]))
    return policy, draft, teacher, events


def test_normalized_l2_excludes_grippers_and_prefix_uses_every_k():
    draft = np.zeros((1, 30, 1, 32, 1), dtype=np.float32)
    teachers = np.zeros((2, 30, 1, 32, 1), dtype=np.float32)
    teachers[:, 28:30] = 100.0
    teachers[1, 0, 0, 20, 0] = np.sqrt(14.0)

    distances = normalized_l2_step_distances(draft, teachers)

    assert distances.shape == (2, 32)
    assert distances[0].max() == 0.0
    assert np.isclose(distances[1, 20], 1.0)
    assert longest_safe_prefix(distances, threshold=0.5) == 16


def test_bounded_endpoint_repair_changes_only_continuous_prefix():
    draft = np.zeros((1, 30, 2, 16, 1), dtype=np.float32)
    endpoint = np.ones_like(draft)
    draft[:, 28:30] = -1.0

    candidate, max_rms = bounded_endpoint_repair(
        draft,
        endpoint,
        prefix_len=16,
        strength=1.0,
        max_step_rms=0.1,
    )

    assert np.isclose(max_rms, 0.1)
    assert np.allclose(candidate[:, :14, 0], 0.1)
    assert np.array_equal(candidate[:, :14, 1], draft[:, :14, 1])
    assert np.array_equal(candidate[:, 14:], draft[:, 14:])


def test_flowguard_near_miss_repair_changes_only_first_fail_window():
    draft = np.zeros((1, 30, 2, 16, 1), dtype=np.float32)
    endpoint = np.ones_like(draft)

    candidate, stats = flowguard_near_miss_repair(
        draft,
        endpoint,
        first_fail_index=5,
        conditioned_frame_count=0,
        threshold=0.15,
        residual=0.16,
        max_strength=0.5,
        max_step_rms=0.15,
    )

    assert np.all(candidate[:, :14, 0, :5] == 0)
    assert np.all(candidate[:, :14, 0, 5:16] > 0)
    assert np.all(candidate[:, :14, 1] == 0)
    assert np.all(candidate[:, 14:] == 0)
    assert candidate[0, 0, 0, 5, 0] > candidate[0, 0, 0, 15, 0]
    assert np.isclose(stats["flowguard_dynamic_strength"], 0.09375)
    assert stats["flowguard_window_start"] == 5
    assert stats["flowguard_window_end"] == 16


def test_flowguard_probe_classifies_near_miss_and_clear_fail():
    response = {
        "distances": np.full((2, 2, 16), 0.16, dtype=np.float32),
        "accepted_prefix_before_gripper": 0,
        "conditioned_frame_count": 0,
    }
    near = classify_flowguard_probe(
        response,
        threshold=0.15,
        repair_threshold=0.18,
        action_per_frame=16,
        horizon=32,
    )
    clear = classify_flowguard_probe(
        response,
        threshold=0.15,
        repair_threshold=0.155,
        action_per_frame=16,
        horizon=32,
    )

    assert near["flowguard_class"] == "near_miss"
    assert near["flowguard_first_fail_index"] == 0
    assert clear["flowguard_class"] == "clear_fail"


def test_flowguard_first_probe_pass_does_not_use_k2_prefix():
    response = {
        "distances": np.array(
            [
                [[[0.01] * 16, [0.20] * 16]],
                [[[0.20] * 16, [0.20] * 16]],
            ],
            dtype=np.float32,
        ).reshape(2, 2, 16),
        "conditioned_frame_count": 0,
        "accepted_prefix_before_gripper": 0,
        "accepted_prefix": 0,
        "gripper_phase_agreement_by_tau": [1.0, 1.0],
        "gripper_switch_indices_by_tau": [None, None],
    }

    result = classify_flowguard_probe(
        response,
        threshold=0.15,
        repair_threshold=0.18,
        action_per_frame=16,
        horizon=32,
    )

    assert result["flowguard_class"] == "pass"
    assert result["flowguard_first_probe_prefix"] == 16
    assert result["flowguard_k1_false_accept"] is True


def test_flowguard_first_probe_phase_risk_is_ambiguous():
    response = {
        "distances": np.full((2, 2, 16), 0.01, dtype=np.float32),
        "conditioned_frame_count": 0,
        "accepted_prefix": 32,
        "gripper_phase_agreement_by_tau": [0.75, 1.0],
        "gripper_switch_indices_by_tau": [None, None],
    }

    result = classify_flowguard_probe(
        response,
        threshold=0.15,
        repair_threshold=0.18,
        action_per_frame=16,
        horizon=32,
    )

    assert result["flowguard_class"] == "ambiguous"
    assert result["flowguard_reason"] == "phase_risk"


def test_gripper_phase_snap_changes_only_small_majority_disagreement():
    latent = np.zeros((1, 30, 2, 16, 1), dtype=np.float32)
    latent[:, 28:30] = -1.0
    draft_phase = np.zeros((2, 2, 16), dtype=bool)
    draft_phase[:, 0, 5:] = True
    draft_phase[:, 1] = True
    latent[0, 28:30, :, :, 0] = np.where(draft_phase, 1.0, -1.0)
    teacher_phase = draft_phase.copy()
    teacher_phase[:, 0, 4] = True

    repaired = gripper_phase_snap_repair(
        latent,
        np.stack((teacher_phase, teacher_phase)),
        teacher_phase[None],
        draft_phase,
        previous_phase=np.array([False, False]),
    )

    assert repaired is not None
    candidate, edit_count = repaired
    assert edit_count == 2
    assert np.array_equal(candidate[:, :28], latent[:, :28])
    assert np.array_equal(candidate[:, 28:30, 0, 4, 0], np.ones((1, 2)))
    unchanged = np.ones((2, 2, 16), dtype=bool)
    unchanged[:, 0, 4] = False
    assert np.array_equal(
        candidate[0, 28:30, :, :, 0][unchanged],
        latent[0, 28:30, :, :, 0][unchanged],
    )


def test_gripper_phase_snap_rejects_multi_transition_pulse():
    latent = np.zeros((1, 30, 2, 16, 1), dtype=np.float32)
    draft_phase = np.zeros((2, 2, 16), dtype=bool)
    pulse = draft_phase.copy()
    pulse[:, 0, 0] = True

    assert gripper_phase_snap_repair(
        latent,
        np.stack((pulse, pulse)),
        pulse[None],
        draft_phase,
        previous_phase=np.array([False, False]),
    ) is None


def test_gripper_phase_snap_respects_conditioned_frame_offset():
    latent = np.zeros((1, 30, 3, 16, 1), dtype=np.float32)
    latent[:, 28:30] = -1.0
    draft_phase = np.zeros((2, 2, 16), dtype=bool)
    teacher_phase = draft_phase.copy()
    teacher_phase[:, 1, -1] = True

    repaired = gripper_phase_snap_repair(
        latent,
        np.stack((teacher_phase, teacher_phase)),
        teacher_phase[None],
        draft_phase,
        frame_offset=1,
    )

    assert repaired is not None
    candidate, edit_count = repaired
    assert edit_count == 2
    assert np.array_equal(candidate[:, :, 0], latent[:, :, 0])
    assert np.array_equal(candidate[0, 28:30, 2, -1, 0], np.ones(2))


def test_gripper_switch_finds_first_transition_not_only_endpoints():
    action = _action()
    action[7, 0, 4:8] = 1.0

    assert first_gripper_switch(action) == 4


def test_gripper_switch_detects_chunk_boundary_against_executed_phase():
    action = _action()

    assert first_gripper_switch(action, previous=np.array([1.0, 0.0])) == 0


def test_first_round_is_full_and_full_cache_update_becomes_anchor():
    events = []
    draft = _FakeModel("draft", events)
    teacher = _FakeModel("teacher", events)
    policy = RealtimeFlashPolicy(draft, teacher, rng=np.random.default_rng(3))

    policy.infer({"reset": True, "prompt": "test task"})
    response = policy.infer(_action_request())

    assert response["action_source"] == "teacher_full"
    assert response["full_reason"] == "initial"
    assert not [call for call in teacher.calls if call["kind"] == "verify"]
    assert [call["kind"] for call in draft.calls] == ["reset", "prime"]

    policy.infer(_cache_request("anchor", response["action"]))

    assert draft.cache == ["anchor"]
    assert teacher.cache == ["anchor"]
    assert policy.teacher_anchor_frame_st_id == 2
    assert not policy.pending_teacher_cache_updates


def test_every_flash_verifies_against_last_full_and_queues_teacher_update():
    policy, draft, teacher, _ = _anchored_policy(verify_results=(32, 32))

    flash = policy.infer(_action_request())
    first_verify = [call for call in teacher.calls if call["kind"] == "verify"][-1]

    assert flash["action_source"] == "draft_flash"
    assert flash["accepted_prefix"] == 32
    assert first_verify["cache_before"] == ("anchor",)
    assert first_verify["frame_before"] == 2
    assert first_verify["request"]["frame_st_id"] == 2
    assert first_verify["request"]["tau_timesteps"] == (50.0, 100.0)
    assert first_verify["request"]["previous_gripper"].tolist() == [-1.0, -1.0]
    assert first_verify["request"]["verify_noise"].shape == (1, 30, 2, 16, 1)
    assert teacher.cache == ["anchor"]
    assert teacher.frame_st_id == 2

    policy.infer(_cache_request("flash-1", flash["action"]))
    assert draft.cache == ["anchor", "flash-1"]
    assert teacher.cache == ["anchor"]
    assert len(policy.pending_teacher_cache_updates) == 1

    policy.infer(_action_request())
    second_verify = [call for call in teacher.calls if call["kind"] == "verify"][-1]
    assert second_verify["cache_before"] == ("anchor",)
    assert second_verify["frame_before"] == 2
    assert second_verify["request"]["frame_st_id"] == 4


def test_periodic_full_replays_pending_teacher_updates_in_order():
    policy, _, teacher, _ = _anchored_policy(
        pf_interval=2, verify_results=(32, 32)
    )

    flash_1 = policy.infer(_action_request())
    policy.infer(_cache_request("flash-1", flash_1["action"]))
    flash_2 = policy.infer(_action_request())
    policy.infer(_cache_request("flash-2", flash_2["action"]))
    full = policy.infer(_action_request())

    assert full["action_source"] == "teacher_full"
    assert full["full_reason"] == "periodic"
    teacher_actions = [call for call in teacher.calls if call["kind"] == "action"]
    assert teacher_actions[-1]["cache_before"] == ("anchor", "flash-1", "flash-2")
    assert teacher.cache == ["anchor", "flash-1", "flash-2"]
    assert not policy.pending_teacher_cache_updates

    policy.infer(_cache_request("full-2", full["action"]))
    assert teacher.cache == ["anchor", "flash-1", "flash-2", "full-2"]
    assert policy.teacher_anchor_frame_st_id == 8


def test_zero_pf_disables_periodic_refresh():
    policy, _, _, _ = _anchored_policy(pf_interval=0, verify_results=(32, 32))

    first = policy.infer(_action_request())
    policy.infer(_cache_request("flash-1", first["action"]))
    second = policy.infer(_action_request())

    assert first["action_source"] == "draft_flash"
    assert second["action_source"] == "draft_flash"


def test_cumulative_flow_budget_triggers_next_full_and_resets():
    distances = np.full((2, 2, 16), 0.04, dtype=np.float32)
    events = []
    draft = _FakeModel("draft", events)
    teacher = _FakeModel(
        "teacher",
        events,
        verify_results=(
            {"accepted_prefix": 32, "distances": distances},
            {"accepted_prefix": 32, "distances": distances},
        ),
    )
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=0,
        flow_budget_threshold=0.07,
        rng=np.random.default_rng(7),
    )
    policy.infer({"reset": True, "prompt": "test task"})
    first = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", first["action"]))

    flash_1 = policy.infer(_action_request())
    policy.infer(_cache_request("flash-1", flash_1["action"]))
    flash_2 = policy.infer(_action_request())
    policy.infer(_cache_request("flash-2", flash_2["action"]))
    full = policy.infer(_action_request())

    assert flash_1["action_source"] == "draft_flash"
    assert flash_2["action_source"] == "draft_flash"
    assert full["action_source"] == "teacher_full"
    assert full["full_reason"] == "flow_budget"
    assert policy.flow_error_budget == 0.0


def test_flow_budget_accumulates_only_during_sustained_low_motion(tmp_path):
    distances = np.full((2, 2, 16), 0.1, dtype=np.float32)
    events = []
    draft = _FakeModel("draft", events, video_motion_global=0.4)
    teacher = _FakeModel(
        "teacher",
        events,
        verify_results=tuple(
            {"accepted_prefix": 32, "distances": distances} for _ in range(4)
        ),
    )
    log_path = tmp_path / "metrics.jsonl"
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=0,
        flow_budget_threshold=0.15,
        flow_budget_motion_ceiling=0.5,
        log_path=log_path,
        rng=np.random.default_rng(7),
    )
    policy.infer({"reset": True, "prompt": "test task"})
    initial = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", initial["action"]))

    flash_1 = policy.infer(_action_request())
    policy.infer(_cache_request("low-1", flash_1["action"]))
    assert np.isclose(policy.flow_error_budget, 0.1)

    draft.video_motion_global = 0.7
    flash_2 = policy.infer(_action_request())
    policy.infer(_cache_request("high-reset", flash_2["action"]))
    assert policy.flow_error_budget == 0.0

    draft.video_motion_global = 0.4
    for tag in ("low-2", "low-3"):
        flash = policy.infer(_action_request())
        policy.infer(_cache_request(tag, flash["action"]))
    full = policy.infer(_action_request())

    flash_records = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if json.loads(line).get("source") == "draft_flash"
    ]
    assert flash_records[1]["flow_budget_motion_reset"] is True
    assert flash_records[1]["flow_budget_charge"] == 0.0
    assert full["full_reason"] == "flow_budget"


def test_invalid_motion_telemetry_does_not_commit_flash_state():
    events = []
    draft = _FakeModel("draft", events, video_motion_global=np.nan)
    teacher = _FakeModel("teacher", events)
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=0,
        flow_budget_threshold=0.4,
        flow_budget_motion_ceiling=0.5,
        rng=np.random.default_rng(7),
    )
    policy.infer({"reset": True, "prompt": "test task"})
    initial = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", initial["action"]))
    round_id = policy.round_id

    with np.testing.assert_raises(ValueError):
        policy.infer(_action_request())

    assert policy.pending_cache_source is None
    assert policy.flash_rounds_since_full == 0
    assert policy.round_id == round_id

    draft.video_motion_global = 0.4
    response = policy.infer(_action_request())
    assert response["action_source"] == "draft_flash"


def test_second_flow_budget_refresh_runs_two_teacher_rounds():
    distances = np.full((2, 2, 16), 0.04, dtype=np.float32)
    events = []
    draft = _FakeModel("draft", events)
    teacher = _FakeModel(
        "teacher",
        events,
        verify_results=(
            {"accepted_prefix": 32, "distances": distances},
            {"accepted_prefix": 32, "distances": distances},
        ),
    )
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=0,
        flow_budget_threshold=0.03,
        flow_budget_burst_after=2,
        flow_budget_burst_rounds=2,
        rng=np.random.default_rng(7),
    )
    policy.infer({"reset": True, "prompt": "test task"})
    initial = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", initial["action"]))

    flash_1 = policy.infer(_action_request())
    policy.infer(_cache_request("flash-1", flash_1["action"]))
    refresh_1 = policy.infer(_action_request())
    policy.infer(_cache_request("refresh-1", refresh_1["action"]))
    flash_2 = policy.infer(_action_request())
    policy.infer(_cache_request("flash-2", flash_2["action"]))
    refresh_2 = policy.infer(_action_request())
    policy.infer(_cache_request("refresh-2", refresh_2["action"]))
    burst = policy.infer(_action_request())

    assert refresh_1["full_reason"] == "flow_budget"
    assert refresh_2["full_reason"] == "flow_budget"
    assert burst["full_reason"] == "flow_budget_burst"
    assert policy.flow_budget_refresh_count == 2
    assert policy.teacher_burst_rounds_left == 0


def test_flow_budget_burst_limit_allows_only_one_episode_burst():
    distances = np.full((2, 2, 16), 0.04, dtype=np.float32)
    events = []
    draft = _FakeModel("draft", events)
    teacher = _FakeModel(
        "teacher",
        events,
        verify_results=tuple(
            {"accepted_prefix": 32, "distances": distances} for _ in range(3)
        ),
    )
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=0,
        flow_budget_threshold=0.03,
        flow_budget_burst_after=2,
        flow_budget_burst_rounds=2,
        flow_budget_burst_limit=1,
        rng=np.random.default_rng(7),
    )
    policy.infer({"reset": True, "prompt": "test task"})
    initial = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", initial["action"]))

    full_reasons = []
    for crossing in range(3):
        flash = policy.infer(_action_request())
        policy.infer(_cache_request(f"flash-{crossing}", flash["action"]))
        refresh = policy.infer(_action_request())
        full_reasons.append(refresh["full_reason"])
        policy.infer(_cache_request(f"refresh-{crossing}", refresh["action"]))
        if crossing == 1:
            burst = policy.infer(_action_request())
            full_reasons.append(burst["full_reason"])
            policy.infer(_cache_request("burst", burst["action"]))

    after_limit = policy.infer(_action_request())

    assert full_reasons == [
        "flow_budget",
        "flow_budget",
        "flow_budget_burst",
        "flow_budget",
    ]
    assert policy.flow_budget_bursts_used == 1
    assert after_limit["action_source"] == "draft_flash"


def test_teacher_reconstructed_gripper_switch_forces_replan():
    policy, _, _, _ = _anchored_policy(
        pf_interval=10,
        verify_results=({
            "accepted_prefix": 0,
            "accepted_prefix_before_gripper": 32,
            "gripper_force_teacher": True,
        },),
    )

    rejected = policy.infer(_action_request())
    full = policy.infer(_action_request())

    assert rejected["replan"] is True
    assert rejected["fallback_reason"] == "teacher_gripper_switch"
    assert full["action_source"] == "teacher_full"
    assert full["full_reason"] == "teacher_gripper_switch"


def test_teacher_reconstructed_gripper_switch_can_be_diagnostic_only():
    events = []
    draft = _FakeModel("draft", events)
    teacher = _FakeModel(
        "teacher",
        events,
        verify_results=({
            "accepted_prefix": 0,
            "accepted_prefix_before_gripper": 32,
            "gripper_force_teacher": True,
        },),
    )
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=10,
        teacher_gripper_fallback=False,
        rng=np.random.default_rng(7),
    )
    policy.infer({"reset": True, "prompt": "test task"})
    first = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", first["action"]))

    accepted = policy.infer(_action_request())

    assert accepted["action_source"] == "draft_flash"
    assert accepted["accepted_prefix"] == 32
    assert accepted["replan"] is False


def test_gripper_consensus_partial_prefix_schedules_teacher_window():
    events = []
    draft = _FakeModel("draft", events, action=_action(switch_step=20))
    teacher = _FakeModel(
        "teacher",
        events,
        verify_results=({
            "accepted_prefix": 16,
            "gripper_consensus_prefix": 16,
            "gripper_consensus_failure_index": 20,
        },),
    )
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=0,
        gripper_consensus=True,
        gripper_full_window=2,
        rng=np.random.default_rng(7),
    )
    policy.infer({"reset": True, "prompt": "test task"})
    first = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", first["action"]))

    partial = policy.infer(_action_request())
    policy.infer(_cache_request("partial", partial["action"]))
    teacher_1 = policy.infer(_action_request())
    policy.infer(_cache_request("teacher-1", teacher_1["action"]))
    teacher_2 = policy.infer(_action_request())

    assert partial["accepted_prefix"] == 16
    assert partial["fallback_reason"] == "gripper_consensus"
    assert teacher_1["full_reason"] == "gripper_consensus"
    assert teacher_2["full_reason"] == "gripper_phase_burst"


def test_gripper_consensus_makes_server_prefix_authoritative():
    events = []
    draft = _FakeModel("draft", events, action=_action(switch_step=20))
    teacher = _FakeModel(
        "teacher",
        events,
        verify_results=({
            "accepted_prefix": 32,
            "gripper_consensus_prefix": 32,
            "gripper_force_teacher": True,
        },),
    )
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=10,
        gripper_consensus=True,
        rng=np.random.default_rng(7),
    )
    policy.infer({"reset": True, "prompt": "test task"})
    first = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", first["action"]))

    accepted = policy.infer(_action_request())
    verify_call = [call for call in teacher.calls if call["kind"] == "verify"][-1]

    assert verify_call["request"]["gripper_consensus"] is True
    assert accepted["action_source"] == "draft_flash"
    assert accepted["accepted_prefix"] == 32
    assert accepted["replan"] is False
    assert "fallback_reason" not in accepted


def test_flash_logs_distances_without_changing_acceptance(tmp_path):
    events = []
    draft = _FakeModel("draft", events)
    teacher = _FakeModel(
        "teacher",
        events,
        verify_results=({
            "accepted_prefix": 32,
            "distances": np.zeros((2, 2, 16), dtype=np.float32),
            "prefix_by_tau": [32, 32],
            "tau_timesteps": [50.0, 100.0],
            "gripper_switch_indices_by_tau": [None, 7],
            "gripper_phase_agreement_by_tau": [1.0, 0.75],
            "draft_gripper_switch_index": 7,
        },),
    )
    log_path = tmp_path / "metrics.jsonl"
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=10,
        log_path=log_path,
        rng=np.random.default_rng(7),
    )
    policy.infer({"reset": True, "prompt": "test task"})
    first = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", first["action"]))

    accepted = policy.infer(_action_request())
    record = json.loads(log_path.read_text().splitlines()[-1])

    assert accepted["accepted_prefix"] == 32
    assert record["prefix_by_tau"] == [32, 32]
    assert record["tau_timesteps"] == [50.0, 100.0]
    assert np.asarray(record["verify_distances"]).shape == (2, 2, 16)
    assert record["gripper_switch_indices_by_tau"] == [None, 7]
    assert record["gripper_phase_agreement_by_tau"] == [1.0, 0.75]
    assert record["draft_gripper_switch_index"] == 7
    assert record["video_motion_stats"]["top_relative"] == 4.0


def test_executed_draft_cache_update_logs_delayed_video_error(tmp_path):
    events = []
    draft = _FakeModel("draft", events)
    teacher = _FakeModel("teacher", events)
    log_path = tmp_path / "metrics.jsonl"
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=10,
        log_path=log_path,
        rng=np.random.default_rng(7),
    )
    policy.infer({"reset": True, "prompt": "test task"})
    first = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", first["action"]))
    flash = policy.infer(_action_request())
    policy.infer(_cache_request("flash", flash["action"]))
    record = json.loads(log_path.read_text().splitlines()[-1])

    draft_cache = [call for call in draft.calls if call["kind"] == "cache"][-1]
    draft_action = [call for call in draft.calls if call["kind"] == "action"][-1]
    assert draft_action["request"]["track_video_prediction"] is True
    assert draft_cache["request"]["compare_video_prediction"] is True
    assert record["delayed_video_error"]["latent_nrmse"] == 0.25


def test_persistent_delayed_error_runs_two_teacher_rounds(tmp_path):
    events = []
    draft = _FakeModel("draft", events, delayed_errors=(0.6, 0.57))
    teacher = _FakeModel("teacher", events)
    log_path = tmp_path / "metrics.jsonl"
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=20,
        delayed_error_threshold=0.55,
        delayed_error_consecutive=2,
        delayed_error_teacher_rounds=2,
        log_path=log_path,
        rng=np.random.default_rng(7),
    )
    policy.infer({"reset": True, "prompt": "test task"})
    initial = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", initial["action"]))

    flash_1 = policy.infer(_action_request())
    policy.infer(_cache_request("flash-1", flash_1["action"]))
    assert policy.delayed_error_streak == 1
    flash_2 = policy.infer(_action_request())
    policy.infer(_cache_request("flash-2", flash_2["action"]))

    full_1 = policy.infer(_action_request())
    policy.infer(_cache_request("recovery-1", full_1["action"]))
    full_2 = policy.infer(_action_request())

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    trigger = next(record for record in records if record.get("delayed_error_triggered"))
    assert trigger["delayed_error_streak"] == 2
    assert trigger["delayed_error_teacher_rounds_left"] == 2
    assert full_1["full_reason"] == "delayed_video_error"
    assert full_2["full_reason"] == "delayed_video_error_burst"
    assert policy.delayed_error_streak == 0
    assert policy.delayed_error_teacher_rounds_left == 0


def test_low_delayed_error_resets_recovery_streak():
    events = []
    draft = _FakeModel("draft", events, delayed_errors=(0.6, 0.4, 0.6))
    teacher = _FakeModel("teacher", events)
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=20,
        delayed_error_threshold=0.55,
        delayed_error_consecutive=2,
        rng=np.random.default_rng(7),
    )
    policy.infer({"reset": True, "prompt": "test task"})
    action = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", action["action"]))
    for tag in ("flash-1", "flash-2", "flash-3"):
        action = policy.infer(_action_request())
        policy.infer(_cache_request(tag, action["action"]))

    assert policy.delayed_error_streak == 1
    assert policy.delayed_error_trigger_count == 0
    assert policy.delayed_error_teacher_rounds_left == 0


def test_delayed_recovery_includes_an_already_scheduled_flow_refresh():
    events = []
    distances = np.zeros((2, 32), dtype=np.float32)
    verify_results = (
        {"accepted_prefix": 32, "distances": distances + 0.01},
        {"accepted_prefix": 32, "distances": distances + 0.1},
    )
    draft = _FakeModel("draft", events, delayed_errors=(0.6, 0.57))
    teacher = _FakeModel("teacher", events, verify_results=verify_results)
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=20,
        flow_budget_threshold=0.1,
        delayed_error_threshold=0.55,
        delayed_error_consecutive=2,
        delayed_error_teacher_rounds=2,
        rng=np.random.default_rng(7),
    )
    policy.infer({"reset": True, "prompt": "test task"})
    initial = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", initial["action"]))
    for tag in ("flash-1", "flash-2"):
        flash = policy.infer(_action_request())
        policy.infer(_cache_request(tag, flash["action"]))

    full_1 = policy.infer(_action_request())
    policy.infer(_cache_request("recovery-1", full_1["action"]))
    full_2 = policy.infer(_action_request())
    policy.infer(_cache_request("recovery-2", full_2["action"]))
    next_action = policy.infer(_action_request())

    assert full_1["full_reason"] == "flow_budget"
    assert full_2["full_reason"] == "delayed_video_error_burst"
    assert next_action["action_source"] == "draft_flash"


def test_video_motion_gate_replans_before_action_verification(tmp_path):
    events = []
    draft = _FakeModel("draft", events, video_motion_global=1.3)
    teacher = _FakeModel("teacher", events)
    log_path = tmp_path / "metrics.jsonl"
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=20,
        video_motion_gate_threshold=1.2,
        log_path=log_path,
        rng=np.random.default_rng(7),
    )
    policy.infer({"reset": True, "prompt": "test task"})
    initial = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", initial["action"]))

    response = infer_with_replan(policy, _action_request())

    teacher_verifies = [call for call in teacher.calls if call["kind"] == "verify"]
    record = json.loads(log_path.read_text().splitlines()[-2])
    assert not teacher_verifies
    assert record["source"] == "replan"
    assert record["fallback_reason"] == "video_motion_risk"
    assert record["video_motion_stats"]["global_mean"] == 1.3
    assert response["action_source"] == "teacher_full"
    assert response["full_reason"] == "video_motion_risk"


def test_video_motion_gate_default_off_still_verifies():
    events = []
    draft = _FakeModel("draft", events, video_motion_global=2.0)
    teacher = _FakeModel("teacher", events)
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=20,
        rng=np.random.default_rng(7),
    )
    policy.infer({"reset": True, "prompt": "test task"})
    initial = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", initial["action"]))

    response = policy.infer(_action_request())

    assert response["action_source"] == "draft_flash"
    assert [call for call in teacher.calls if call["kind"] == "verify"]


def test_zero_prefix_replans_same_observation_without_cache_update():
    policy, draft, teacher, _ = _anchored_policy(verify_results=(0,))
    request = _action_request()

    class _RecordingPolicy:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.request_ids = []

        def infer(self, action_request):
            self.request_ids.append(id(action_request))
            return self.wrapped.infer(action_request)

    recording_policy = _RecordingPolicy(policy)
    response = infer_with_replan(recording_policy, request)

    assert recording_policy.request_ids == [id(request), id(request)]
    assert response["action_source"] == "teacher_full"
    assert response["full_reason"] == "zero_prefix"
    assert draft.cache == ["anchor"]
    assert teacher.cache == ["anchor"]
    assert not policy.pending_teacher_cache_updates


def test_zero_prefix_shadow_repair_uses_holdout_and_still_replans(tmp_path):
    events = []
    draft = _FakeModel("draft", events)
    endpoint = np.zeros((1, 30, 2, 16, 1), dtype=np.float32)
    endpoint[:, :14, 0] = 0.2
    teacher = _FakeModel(
        "teacher",
        events,
        verify_results=(
            {
                "accepted_prefix": 0,
                "teacher_endpoint_latent": endpoint,
                "continuous_channels": list(range(14)),
            },
            {"accepted_prefix": 16, "prefix_by_tau": [16, 16]},
        ),
    )
    log_path = tmp_path / "metrics.jsonl"
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=20,
        repair_shadow=True,
        rng=np.random.default_rng(7),
        repair_rng=np.random.default_rng(8),
        log_path=log_path,
    )
    policy.infer({"reset": True, "prompt": "test task"})
    initial = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", initial["action"]))

    response = policy.infer(_action_request())

    verifies = [call for call in teacher.calls if call["kind"] == "verify"]
    record = json.loads(log_path.read_text().splitlines()[-1])
    assert response["replan"] is True
    assert response["fallback_reason"] == "zero_prefix"
    assert len(verifies) == 2
    assert not np.array_equal(
        verifies[0]["request"]["verify_noise"],
        verifies[1]["request"]["verify_noise"],
    )
    assert record["repair_shadow_eligible"] is True
    assert record["repair_shadow_holdout_prefix"] == 16
    assert record["repair_shadow_correction_max_rms"] <= 0.15
    assert policy.pending_cache_source is None


def test_flowguard_shadow_pairs_original_and_repaired_on_probe_b(tmp_path):
    events = []
    draft = _FakeModel("draft", events)
    endpoint = np.zeros((1, 30, 2, 16, 1), dtype=np.float32)
    endpoint[:, :14] = 0.2
    primary = {
        "accepted_prefix": 0,
        "accepted_prefix_before_gripper": 0,
        "conditioned_frame_count": 0,
        "distances": np.full((2, 2, 16), 0.16, dtype=np.float32),
        "teacher_endpoint_latent_first_probe": endpoint,
        "continuous_channels": list(range(14)),
        "gripper_force_teacher": False,
        "gripper_consensus_failure_index": None,
        "draft_gripper_switch_index": None,
        "cross_tau_endpoint_rms": 0.01,
        "cross_tau_direction_cosine": 0.99,
    }
    teacher = _FakeModel(
        "teacher",
        events,
        verify_results=(
            primary,
            {"accepted_prefix": 0, "prefix_by_tau": [0, 0]},
            {"accepted_prefix": 16, "prefix_by_tau": [16, 16]},
        ),
    )
    log_path = tmp_path / "metrics.jsonl"
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=20,
        flowguard_nearmiss_shadow=True,
        rng=np.random.default_rng(7),
        shadow_rng=np.random.default_rng(9),
        log_path=log_path,
    )
    policy.infer({"reset": True, "prompt": "test task"})
    initial = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", initial["action"]))

    response = policy.infer(_action_request())

    verifies = [call for call in teacher.calls if call["kind"] == "verify"]
    record = json.loads(log_path.read_text().splitlines()[-1])
    assert response["replan"] is True
    assert len(verifies) == 3
    assert np.array_equal(
        verifies[1]["request"]["verify_noise"],
        verifies[2]["request"]["verify_noise"],
    )
    assert not np.array_equal(
        verifies[0]["request"]["verify_noise"],
        verifies[1]["request"]["verify_noise"],
    )
    assert record["flowguard_class"] == "near_miss"
    assert record["flowguard_shadow_eligible"] is True
    assert record["flowguard_shadow_rescue"] is True
    assert record["flowguard_original_holdout_prefix"] == 0
    assert record["flowguard_repaired_holdout_prefix"] == 16
    assert record["flowguard_shadow_action_only_forwards"] == 4
    assert policy.pending_cache_source is None


def test_zero_prefix_repair_executes_only_independent_holdout_prefix(tmp_path):
    events = []
    draft = _FakeModel("draft", events)
    endpoint = np.zeros((1, 30, 2, 16, 1), dtype=np.float32)
    endpoint[:, :14, 0] = 0.2
    repaired_action = _action(value=0.3)
    teacher = _FakeModel(
        "teacher",
        events,
        verify_results=(
            {
                "accepted_prefix": 0,
                "teacher_endpoint_latent": endpoint,
                "continuous_channels": list(range(14)),
            },
            {
                "accepted_prefix": 16,
                "prefix_by_tau": [16, 16],
                "stitched_action": repaired_action,
                "distances": np.zeros((2, 2, 16), dtype=np.float32),
            },
        ),
    )
    log_path = tmp_path / "metrics.jsonl"
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=20,
        zero_prefix_repair=True,
        rng=np.random.default_rng(7),
        repair_rng=np.random.default_rng(8),
        log_path=log_path,
    )
    policy.infer({"reset": True, "prompt": "test task"})
    initial = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", initial["action"]))

    response = policy.infer(_action_request())

    verifies = [call for call in teacher.calls if call["kind"] == "verify"]
    record = json.loads(log_path.read_text().splitlines()[-1])
    assert len(verifies) == 2
    assert not np.array_equal(
        verifies[0]["request"]["verify_noise"],
        verifies[1]["request"]["verify_noise"],
    )
    assert response["action_source"] == "draft_flash"
    assert response["accepted_prefix"] == 16
    assert np.array_equal(response["action"], repaired_action[:, :1])
    assert record["repair_executed"] is True
    assert record["repair_kind"] == "zero_prefix_endpoint"
    assert record["repair_holdout_prefix"] == 16
    assert record["repair_correction_max_rms"] <= 0.15
    assert policy.pending_cache_source == "flash"


def test_zero_prefix_repair_failed_holdout_keeps_teacher_replan(tmp_path):
    events = []
    draft = _FakeModel("draft", events)
    endpoint = np.zeros((1, 30, 2, 16, 1), dtype=np.float32)
    teacher = _FakeModel(
        "teacher",
        events,
        verify_results=(
            {"accepted_prefix": 0, "teacher_endpoint_latent": endpoint},
            {"accepted_prefix": 0},
        ),
    )
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=20,
        zero_prefix_repair=True,
        rng=np.random.default_rng(7),
        repair_rng=np.random.default_rng(8),
        log_path=tmp_path / "metrics.jsonl",
    )
    policy.infer({"reset": True, "prompt": "test task"})
    initial = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", initial["action"]))

    response = policy.infer(_action_request())

    assert response["replan"] is True
    assert response["fallback_reason"] == "zero_prefix"
    assert policy.pending_cache_source is None


def test_flow_consistent_repair_executes_exact_independent_holdout_prefix(tmp_path):
    events = []
    draft = _FakeModel("draft", events)
    candidate = np.zeros((1, 30, 2, 16, 1), dtype=np.float32)
    candidate[:, :14, 0] = 0.1
    repaired_action = _action(value=0.3)
    teacher = _FakeModel(
        "teacher",
        events,
        verify_results=(
            {
                "accepted_prefix": 0,
                "accepted_prefix_before_gripper": 0,
                "flow_repair_eligible": True,
                "flow_repair_candidate_latent": candidate,
                "flow_repair_stats": {
                    "repair_timestep": 50.0,
                    "midpoint_timestep": 25.0,
                    "construction_forward_count": 1,
                },
            },
            {
                "accepted_prefix": 32,
                "prefix_by_tau": [32, 32],
                "stitched_action": repaired_action,
                "distances": np.zeros((2, 2, 16), dtype=np.float32),
            },
        ),
    )
    log_path = tmp_path / "metrics.jsonl"
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=20,
        flow_consistent_repair=True,
        rng=np.random.default_rng(7),
        repair_rng=np.random.default_rng(8),
        log_path=log_path,
    )
    policy.infer({"reset": True, "prompt": "test task"})
    initial = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", initial["action"]))

    response = policy.infer(_action_request())

    verifies = [call for call in teacher.calls if call["kind"] == "verify"]
    record = json.loads(log_path.read_text().splitlines()[-1])
    assert len(verifies) == 2
    assert verifies[0]["request"]["flow_repair"] is True
    assert not np.array_equal(
        verifies[0]["request"]["verify_noise"],
        verifies[1]["request"]["verify_noise"],
    )
    assert response["action_source"] == "draft_flash"
    assert response["accepted_prefix"] == 32
    assert np.array_equal(response["action"], repaired_action)
    assert record["repair_executed"] is True
    assert record["repair_kind"] == "zero_prefix_flow_rk2"
    assert record["repair_holdout_prefix"] == 32
    assert record["primary_verify_forwards"] == 2
    assert record["repair_construction_forwards"] == 1
    assert record["repair_holdout_forwards"] == 2
    assert record["repair_action_only_forwards"] == 3
    assert record["repair_flow_stats"]["midpoint_timestep"] == 25.0


def test_flow_consistent_repair_nonfinite_state_fails_closed(tmp_path):
    events = []
    teacher = _FakeModel(
        "teacher",
        events,
        verify_results=(
            {
                "accepted_prefix": 0,
                "accepted_prefix_before_gripper": 0,
                "flow_repair_eligible": False,
                "flow_repair_candidate_latent": None,
                "flow_repair_stats": {
                    "construction_forward_count": 1,
                    "rejection_reason": "non_finite_flow_state",
                },
            },
        ),
    )
    log_path = tmp_path / "metrics.jsonl"
    policy = RealtimeFlashPolicy(
        _FakeModel("draft", events),
        teacher,
        pf_interval=20,
        flow_consistent_repair=True,
        log_path=log_path,
    )
    policy.infer({"reset": True, "prompt": "test task"})
    initial = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", initial["action"]))

    response = policy.infer(_action_request())

    record = json.loads(log_path.read_text().splitlines()[-1])
    assert response["replan"] is True
    assert response["fallback_reason"] == "zero_prefix"
    assert record["repair_attempted"] is False
    assert record["repair_construction_forwards"] == 1
    assert record["repair_holdout_forwards"] == 0
    assert record["repair_flow_stats"]["rejection_reason"] == \
        "non_finite_flow_state"


def test_zero_prefix_repair_rejects_unaligned_prefix_length():
    with pytest.raises(ValueError, match="align to action frames"):
        RealtimeFlashPolicy(
            _FakeModel("draft", []),
            _FakeModel("teacher", []),
            zero_prefix_repair=True,
            repair_prefix_len=8,
        )


def test_shadow_probe_does_not_consume_executable_repair_rng():
    events = []
    teacher = _FakeModel(
        "teacher",
        events,
        verify_results=(
            {"accepted_prefix": 0},
            {"accepted_prefix": 0},
        ),
    )
    policy = RealtimeFlashPolicy(
        _FakeModel("draft", events),
        teacher,
        repair_shadow=True,
        repair_rng=np.random.default_rng(8),
        shadow_rng=np.random.default_rng(9),
    )
    latent = np.zeros((1, 30, 2, 16, 1), dtype=np.float32)
    response = {"teacher_endpoint_latent": latent.copy()}

    policy._shadow_repair_probe(_action_request(), latent, response, 32)
    holdout_request = _action_request()
    holdout_request["flow_repair"] = True
    policy._independent_repair_verify(holdout_request, latent, 32)

    expected = np.random.default_rng(8).standard_normal(latent.shape).astype(
        np.float32
    )
    assert np.array_equal(teacher.calls[1]["request"]["verify_noise"], expected)
    assert teacher.calls[1]["request"]["flow_repair"] is False


@pytest.mark.parametrize("holdout_prefix", [16, 32])
def test_gripper_phase_repair_executes_independently_verified_prefix(
    tmp_path, holdout_prefix
):
    events = []
    draft_action = _action(switch_step=5, value=0.25)
    draft_action[15, 0, 5:] = 1.0
    draft = _FakeModel("draft", events, action=draft_action)
    draft_phase = np.zeros((2, 2, 16), dtype=bool)
    draft_phase[:, 0, 5:] = True
    draft_phase[:, 1] = True
    draft.action_latent[:, 28:30] = np.where(
        draft_phase[None, :, :, :, None], 1.0, -1.0
    )
    teacher_phase = draft_phase.copy()
    teacher_phase[:, 0, 4] = True
    repaired_action = draft_action.copy()
    repaired_action[[7, 15], 0, 4] = 1.0
    teacher = _FakeModel(
        "teacher",
        events,
        verify_results=(
            {
                "accepted_prefix": 0,
                "accepted_prefix_before_gripper": 32,
                "gripper_consensus_failure_index": 4,
                "gripper_phase_by_tau": np.stack(
                    (teacher_phase, teacher_phase)
                ).tolist(),
                "draft_gripper_phase": draft_phase.tolist(),
                "gripper_channels": [28, 29],
            },
            {
                "accepted_prefix": 32,
                "accepted_prefix_before_gripper": 32,
                "gripper_phase_by_tau": teacher_phase[None].tolist(),
            },
            {
                "accepted_prefix": holdout_prefix,
                "accepted_prefix_before_gripper": 32,
                "prefix_by_tau": [holdout_prefix, holdout_prefix],
                "stitched_action": repaired_action,
                "distances": np.zeros((2, 2, 16), dtype=np.float32),
            },
        ),
    )
    log_path = tmp_path / "metrics.jsonl"
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=20,
        gripper_consensus=True,
        gripper_repair=True,
        rng=np.random.default_rng(7),
        repair_rng=np.random.default_rng(8),
        log_path=log_path,
    )
    policy.infer({"reset": True, "prompt": "test task"})
    initial = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", initial["action"]))

    response = policy.infer(_action_request())

    verifies = [call for call in teacher.calls if call["kind"] == "verify"]
    record = json.loads(log_path.read_text().splitlines()[-1])
    assert len(verifies) == 3
    assert verifies[0]["request"]["return_gripper_phase"] is True
    assert verifies[1]["request"]["tau_timesteps"] == (75.0,)
    assert np.array_equal(
        verifies[0]["request"]["verify_noise"],
        verifies[1]["request"]["verify_noise"],
    )
    assert not np.array_equal(
        verifies[0]["request"]["verify_noise"],
        verifies[2]["request"]["verify_noise"],
    )
    assert response["action_source"] == "draft_flash"
    assert response["accepted_prefix"] == holdout_prefix
    expected_action = repaired_action if holdout_prefix == 32 \
        else repaired_action[:, :1]
    assert np.array_equal(response["action"], expected_action)
    assert record["repair_executed"] is True
    assert record["repair_kind"] == "gripper_phase_snap"
    assert record["repair_holdout_prefix"] == holdout_prefix
    assert record["repair_phase_edit_count"] == 2
    assert record["repair_construction_forwards"] == 1
    assert record["repair_holdout_forwards"] == 2
    assert policy.force_full_reason is None


def test_gripper_repair_holdout_cannot_bypass_final_gripper_rejection():
    events = []
    draft = _FakeModel("draft", events)
    draft_phase = np.zeros((2, 2, 16), dtype=bool)
    teacher_phase = draft_phase.copy()
    teacher_phase[:, 1, -1] = True
    teacher = _FakeModel(
        "teacher",
        events,
        verify_results=(
            {
                "accepted_prefix": 0,
                "accepted_prefix_before_gripper": 32,
                "gripper_consensus_failure_index": 31,
                "gripper_phase_by_tau": np.stack(
                    (teacher_phase, teacher_phase)
                ).tolist(),
                "draft_gripper_phase": draft_phase.tolist(),
                "gripper_channels": [28, 29],
            },
            {
                "accepted_prefix": 32,
                "accepted_prefix_before_gripper": 32,
                "gripper_phase_by_tau": teacher_phase[None].tolist(),
            },
            {
                "accepted_prefix": 0,
                "accepted_prefix_before_gripper": 32,
                "gripper_consensus_failure_index": 31,
            },
        ),
    )
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=20,
        gripper_consensus=True,
        gripper_repair=True,
        teacher_gripper_fallback=False,
        rng=np.random.default_rng(7),
        repair_rng=np.random.default_rng(8),
    )
    policy.infer({"reset": True, "prompt": "test task"})
    initial = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", initial["action"]))

    response = policy.infer(_action_request())

    assert response["replan"] is True
    assert response["fallback_reason"] == "gripper_consensus"
    assert response["verified_prefix"] == 0
    assert policy.pending_cache_source is None


def test_gripper_repair_early_exit_counts_only_tiebreak_forward(tmp_path):
    events = []
    teacher = _FakeModel(
        "teacher",
        events,
        verify_results=(
            {
                "accepted_prefix": 0,
                "accepted_prefix_before_gripper": 32,
                "gripper_consensus_failure_index": 4,
            },
            {
                "accepted_prefix": 16,
                "accepted_prefix_before_gripper": 16,
            },
        ),
    )
    log_path = tmp_path / "metrics.jsonl"
    policy = RealtimeFlashPolicy(
        _FakeModel("draft", events),
        teacher,
        pf_interval=20,
        gripper_consensus=True,
        gripper_repair=True,
        log_path=log_path,
    )
    policy.infer({"reset": True, "prompt": "test task"})
    initial = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", initial["action"]))

    response = policy.infer(_action_request())

    record = json.loads(log_path.read_text().splitlines()[-1])
    assert response["replan"] is True
    assert response["fallback_reason"] == "gripper_consensus"
    assert record["repair_construction_forwards"] == 1
    assert record["repair_holdout_forwards"] == 0
    assert record["repair_holdout_evaluated"] is False
    assert record["repair_action_only_forwards"] == 1


def test_gripper_repair_requires_consensus():
    with pytest.raises(ValueError, match="requires gripper consensus"):
        RealtimeFlashPolicy(
            _FakeModel("draft", []),
            _FakeModel("teacher", []),
            gripper_repair=True,
        )


def test_gripper_repair_requires_exactly_two_primary_probes():
    with pytest.raises(ValueError, match="exactly two"):
        RealtimeFlashPolicy(
            _FakeModel("draft", []),
            _FakeModel("teacher", []),
            gripper_consensus=True,
            gripper_repair=True,
            tau_timesteps=(50.0, 75.0, 100.0),
        )


def test_direct_response_is_not_retried():
    class _DirectPolicy:
        def __init__(self):
            self.calls = 0

        def infer(self, request):
            self.calls += 1
            return {"action": _action()}

    direct = _DirectPolicy()
    response = infer_with_replan(direct, _action_request())

    assert direct.calls == 1
    assert response["action"].shape == (16, 2, 16)


def test_gripper_cut_executes_safe_frame_then_forces_full():
    policy, _, teacher, _ = _anchored_policy(
        pf_interval=10,
        draft_action=_action(switch_step=20),
        verify_results=(32,),
    )

    flash = policy.infer(_action_request())

    assert len([call for call in teacher.calls if call["kind"] == "verify"]) == 1
    assert flash["accepted_prefix"] == 16
    assert flash["action"].shape == (16, 1, 16)
    assert flash["fallback_reason"] == "gripper_switch"

    policy.infer(_cache_request("gripper-cut", flash["action"]))
    full = policy.infer(_action_request())
    teacher_actions = [call for call in teacher.calls if call["kind"] == "action"]

    assert full["action_source"] == "teacher_full"
    assert full["full_reason"] == "gripper_switch"
    assert teacher_actions[-1]["cache_before"] == ("anchor", "gripper-cut")


def test_gripper_full_window_two_runs_two_teacher_rounds():
    events = []
    draft = _FakeModel("draft", events, action=_action(switch_step=20))
    teacher = _FakeModel("teacher", events, verify_results=(32,))
    policy = RealtimeFlashPolicy(
        draft,
        teacher,
        pf_interval=10,
        gripper_full_window=2,
        rng=np.random.default_rng(7),
    )
    policy.infer({"reset": True, "prompt": "test task"})
    initial = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", initial["action"]))

    flash = policy.infer(_action_request())
    policy.infer(_cache_request("gripper-cut", flash["action"]))
    full_1 = policy.infer(_action_request())
    policy.infer(_cache_request("phase-1", full_1["action"]))
    full_2 = policy.infer(_action_request())

    assert full_1["full_reason"] == "gripper_switch"
    assert full_2["full_reason"] == "gripper_phase_burst"
    assert policy.gripper_full_rounds_left == 0


def test_reset_clears_pending_updates_and_restores_first_full():
    policy, _, _, _ = _anchored_policy(verify_results=(32,))
    flash = policy.infer(_action_request())
    policy.infer(_cache_request("pending", flash["action"]))
    assert policy.pending_teacher_cache_updates

    policy.infer({"reset": True, "prompt": "new episode"})
    response = policy.infer(_action_request())

    assert not policy.pending_teacher_cache_updates
    assert policy.last_gripper is None
    assert policy.pending_gripper is not None
    assert response["action_source"] == "teacher_full"
    assert response["full_reason"] == "initial"
