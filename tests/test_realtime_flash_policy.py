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
    def __init__(self, role, events, action=None, verify_results=()):
        self.role = role
        self.events = events
        self.action = _action(value=0.25) if action is None else action
        self.action_latent = np.zeros((1, 30, 2, 16, 1), dtype=np.float32)
        self.verify_results = deque(verify_results)
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
            return {}
        if kind == "verify":
            result = self.verify_results.popleft() if self.verify_results else 32
            return dict(result) if isinstance(result, dict) else {"accepted_prefix": result}
        response = {"action": self.action.copy()}
        if self.role == "draft":
            response["action_latent"] = self.action_latent.copy()
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


def test_gripper_switch_finds_first_transition_not_only_endpoints():
    action = _action()
    action[7, 0, 4:8] = 1.0

    assert first_gripper_switch(action) == 4


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


def test_teacher_reconstructed_gripper_switch_forces_replan():
    policy, _, _, _ = _anchored_policy(
        pf_interval=10,
        verify_results=({
            "accepted_prefix": 32,
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
            "accepted_prefix": 32,
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


def test_reset_clears_pending_updates_and_restores_first_full():
    policy, _, _, _ = _anchored_policy(verify_results=(32,))
    flash = policy.infer(_action_request())
    policy.infer(_cache_request("pending", flash["action"]))
    assert policy.pending_teacher_cache_updates

    policy.infer({"reset": True, "prompt": "new episode"})
    response = policy.infer(_action_request())

    assert not policy.pending_teacher_cache_updates
    assert response["action_source"] == "teacher_full"
    assert response["full_reason"] == "initial"
