from collections import deque
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.robotwin.realtime_flash_policy import (
    RealtimeFlashPolicy,
    first_gripper_switch,
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


def _first_verify_noise(policy, draft, teacher, seed):
    policy.infer({
        "reset": True,
        "prompt": "test task",
        "paired_rng_seed": seed,
    })
    first = policy.infer(_action_request())
    policy.infer(_cache_request("anchor", first["action"]))
    policy.infer(_action_request())
    return next(
        call["request"]["verify_noise"].copy()
        for call in reversed(teacher.calls)
        if call["kind"] == "verify"
    )


def test_paired_rng_reset_replays_verifier_noise_per_episode():
    events = []
    draft = _FakeModel("draft", events)
    teacher = _FakeModel("teacher", events)
    policy = RealtimeFlashPolicy(draft, teacher, pf_interval=20)

    first = _first_verify_noise(policy, draft, teacher, 10003)
    policy.rng.standard_normal((9, 9))
    repeated = _first_verify_noise(policy, draft, teacher, 10003)
    different = _first_verify_noise(policy, draft, teacher, 10004)

    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, different)


def test_adaptive_k_shadow_is_forwarded_without_changing_policy_decision():
    telemetry = {
        "accepted_prefix": 32,
        "adaptive_k_shadow": True,
        "adaptive_k_certificate_kind": "pass",
        "adaptive_k_certificate_matches_full": True,
        "requested_verify_k": 2,
        "effective_verify_k": 2,
    }
    policy, _, teacher, _ = _anchored_policy(
        pf_interval=20, verify_results=(telemetry,)
    )
    policy.adaptive_k_shadow = True
    policy.adaptive_k_distance_threshold = 0.05

    response = policy.infer(_action_request())
    verify_call = next(call for call in teacher.calls if call["kind"] == "verify")

    assert response["accepted_prefix"] == 32
    assert verify_call["request"]["adaptive_k_shadow"] is True
    assert verify_call["request"]["adaptive_k_distance_threshold"] == 0.05


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
