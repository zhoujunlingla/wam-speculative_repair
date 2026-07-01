import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.robotwin.specverify_client_policy import (
    RiskRouterClientPolicy,
    SpecVerifyClientPolicy,
    has_gripper_switch,
    slice_action_prefix,
)


def test_slice_action_prefix_keeps_frame_boundary_chunks():
    action = np.zeros((16, 2, 16), dtype=np.float32)

    assert slice_action_prefix(action, 0).shape == (16, 0, 16)
    assert slice_action_prefix(action, 16).shape == (16, 1, 16)
    assert slice_action_prefix(action, 32).shape == (16, 2, 16)


def test_phase_fallback_detects_gripper_switch():
    action = np.zeros((16, 2, 16), dtype=np.float32)
    assert not has_gripper_switch(action)

    action[7, 0, 0] = 0.0
    action[7, 1, 15] = 1.0
    assert has_gripper_switch(action)


class _FakeWsPolicy:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.calls = []

    def infer(self, obs):
        self.calls.append(dict(obs))
        return {}


def test_compute_kv_cache_is_mirrored_to_draft_and_teacher_after_draft_accept():
    clients = []

    def factory(host, port):
        client = _FakeWsPolicy(host, port)
        clients.append(client)
        return client

    policy = SpecVerifyClientPolicy(
        draft_port=1001,
        teacher_port=1002,
        client_factory=factory,
    )
    policy.last_source = "draft"

    state = np.zeros((16, 1, 16), dtype=np.float32)
    ret = policy.infer({"compute_kv_cache": True, "obs": ["real"], "state": state})

    assert ret == {}
    assert policy.frame_st_id == 1
    assert clients[0].calls[-1]["compute_kv_cache"] is True
    assert clients[1].calls[-1]["compute_kv_cache"] is True


def test_risk_router_world_verify_can_reject_action_accepted_chunk(tmp_path):
    clients = []

    class _WorldFakeClient:
        def __init__(self, role):
            self.role = role
            self.calls = []

        def infer(self, obs):
            self.calls.append(dict(obs))
            if obs.get("verify_action"):
                return {"accepted_prefix": 32, "raw_valid_prefix": 32}
            if obs.get("verify_world_latent"):
                return {"world_pass": False, "world_distance_max": 9.0}
            if obs.get("return_action_latent"):
                action = np.zeros((16, 2, 16), dtype=np.float32)
                action[:, 1, :] = 1.0
                return {
                    "action": action,
                    "action_latent": np.zeros((1, 30, 2, 16, 1), dtype=np.float32),
                    "video_latent": np.zeros((1, 48, 2, 24, 20), dtype=np.float32),
                }
            return {"action": np.full((16, 2, 16), 3.0, dtype=np.float32)}

    def factory(host, port):
        client = _WorldFakeClient("draft" if not clients else "teacher")
        clients.append(client)
        return client

    log_path = tmp_path / "metrics.jsonl"
    policy = RiskRouterClientPolicy(
        draft_port=1,
        teacher_port=2,
        teacher_cache_mode="sync",
        risk_low=0.0,
        risk_high=0.5,
        world_verify_enable=True,
        log_path=str(log_path),
        client_factory=factory,
    )
    policy.frame_st_id = 2
    ret = policy.infer({"obs": ["x"], "state": np.zeros((16, 2, 16), dtype=np.float32)})
    assert ret["action"].shape == (16, 2, 16)
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert records[-1]["source"] == "teacher_world_verify_reject"
    assert records[-1]["verify"]["world_verify"]["world_pass"] is False


def test_risk_router_primes_teacher_instead_of_initial_partial_prefix(tmp_path):
    clients = []

    class _PartialPrefixClient:
        def __init__(self, role):
            self.role = role
            self.calls = []

        def infer(self, obs):
            self.calls.append(dict(obs))
            if obs.get("verify_action"):
                return {"accepted_prefix": 16, "raw_valid_prefix": 16}
            if obs.get("verify_world_latent"):
                return {"world_pass": True, "world_distance_max": 0.0}
            if obs.get("return_action_latent"):
                action = np.zeros((16, 2, 16), dtype=np.float32)
                action[:, 1, :] = 1.0
                return {
                    "action": action,
                    "action_latent": np.zeros((1, 30, 2, 16, 1), dtype=np.float32),
                    "video_latent": np.zeros((1, 48, 2, 24, 20), dtype=np.float32),
                }
            return {"action": np.full((16, 2, 16), 3.0, dtype=np.float32)}

    def factory(host, port):
        client = _PartialPrefixClient("draft" if not clients else "teacher")
        clients.append(client)
        return client

    log_path = tmp_path / "metrics.jsonl"
    policy = RiskRouterClientPolicy(
        draft_port=1,
        teacher_port=2,
        teacher_cache_mode="sync",
        risk_low=0.0,
        risk_high=0.5,
        world_verify_enable=True,
        log_path=str(log_path),
        client_factory=factory,
    )
    ret = policy.infer({"obs": ["x"], "state": np.zeros((16, 2, 16), dtype=np.float32)})

    assert np.all(ret["action"] == 3.0)
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert records[-1]["source"] == "teacher_initial_prime"


def test_risk_router_uses_teacher_for_initial_full_prefix_to_prime_cache(tmp_path):
    clients = []

    class _InitialFullPrefixClient:
        def __init__(self, role):
            self.role = role
            self.calls = []

        def infer(self, obs):
            self.calls.append(dict(obs))
            if obs.get("verify_action"):
                return {"accepted_prefix": 32, "raw_valid_prefix": 32}
            if obs.get("verify_world_latent"):
                return {"world_pass": True, "world_distance_max": 0.0}
            if obs.get("return_action_latent"):
                action = np.zeros((16, 2, 16), dtype=np.float32)
                action[:, 1, :] = 1.0
                return {
                    "action": action,
                    "action_latent": np.zeros((1, 30, 2, 16, 1), dtype=np.float32),
                    "video_latent": np.zeros((1, 48, 2, 24, 20), dtype=np.float32),
                }
            return {"action": np.full((16, 2, 16), 3.0, dtype=np.float32)}

    def factory(host, port):
        client = _InitialFullPrefixClient("draft" if not clients else "teacher")
        clients.append(client)
        return client

    log_path = tmp_path / "metrics.jsonl"
    policy = RiskRouterClientPolicy(
        draft_port=1,
        teacher_port=2,
        teacher_cache_mode="sync",
        risk_low=0.0,
        risk_high=0.5,
        world_verify_enable=True,
        log_path=str(log_path),
        client_factory=factory,
    )
    ret = policy.infer({"obs": ["x"], "state": np.zeros((16, 2, 16), dtype=np.float32)})

    assert np.all(ret["action"] == 3.0)
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert records[-1]["source"] == "teacher_initial_prime"


def test_risk_router_uses_teacher_for_initial_low_risk_to_prime_cache(tmp_path):
    clients = []

    class _InitialLowRiskClient:
        def __init__(self, role):
            self.role = role
            self.calls = []

        def infer(self, obs):
            self.calls.append(dict(obs))
            if obs.get("return_action_latent"):
                return {
                    "action": np.zeros((16, 2, 16), dtype=np.float32),
                    "action_latent": np.zeros((1, 30, 2, 16, 1), dtype=np.float32),
                    "video_latent": np.zeros((1, 48, 2, 24, 20), dtype=np.float32),
                }
            return {"action": np.full((16, 2, 16), 3.0, dtype=np.float32)}

    def factory(host, port):
        client = _InitialLowRiskClient("draft" if not clients else "teacher")
        clients.append(client)
        return client

    log_path = tmp_path / "metrics.jsonl"
    policy = RiskRouterClientPolicy(
        draft_port=1,
        teacher_port=2,
        teacher_cache_mode="sync",
        risk_low=0.25,
        risk_high=0.5,
        world_verify_enable=True,
        log_path=str(log_path),
        client_factory=factory,
    )
    ret = policy.infer({"obs": ["x"], "state": np.zeros((16, 2, 16), dtype=np.float32)})

    assert np.all(ret["action"] == 3.0)
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert records[-1]["source"] == "teacher_initial_prime"


def test_risk_router_repairs_action_reject_before_teacher_fallback(tmp_path):
    clients = []

    class _RepairClient:
        def __init__(self, role):
            self.role = role
            self.calls = []

        def infer(self, obs):
            self.calls.append(dict(obs))
            if obs.get("verify_action"):
                if obs.get("return_repair"):
                    return {
                        "accepted_prefix": 0,
                        "raw_valid_prefix": 0,
                        "repair_action": np.full((16, 2, 16), 2.0, dtype=np.float32),
                        "repair_action_latent": np.full((1, 30, 2, 16, 1), 0.25, dtype=np.float32),
                        "repair_delta_mean": 0.05,
                    }
                return {
                    "accepted_prefix": 32,
                    "raw_valid_prefix": 32,
                    "distance_p95": 0.01,
                }
            if obs.get("verify_world_latent"):
                return {"world_pass": True, "world_score": 0.1}
            if obs.get("return_action_latent"):
                action = np.zeros((16, 2, 16), dtype=np.float32)
                action[:, 1, :] = 1.0
                return {
                    "action": action,
                    "action_latent": np.zeros((1, 30, 2, 16, 1), dtype=np.float32),
                    "video_latent": np.zeros((1, 48, 2, 24, 20), dtype=np.float32),
                }
            return {"action": np.full((16, 2, 16), 3.0, dtype=np.float32)}

    def factory(host, port):
        client = _RepairClient("draft" if not clients else "teacher")
        clients.append(client)
        return client

    log_path = tmp_path / "metrics.jsonl"
    policy = RiskRouterClientPolicy(
        draft_port=1,
        teacher_port=2,
        teacher_cache_mode="sync",
        risk_low=0.0,
        risk_high=0.5,
        world_verify_enable=True,
        repair_enable=True,
        log_path=str(log_path),
        client_factory=factory,
    )
    policy.frame_st_id = 2

    ret = policy.infer({"obs": ["x"], "state": np.zeros((16, 2, 16), dtype=np.float32)})

    assert np.all(ret["action"] == 2.0)
    teacher_verify_calls = [call for call in clients[1].calls if call.get("verify_action")]
    assert len(teacher_verify_calls) == 2
    assert teacher_verify_calls[0]["return_repair"] is True
    np.testing.assert_array_equal(
        teacher_verify_calls[1]["action_latent"],
        np.full((1, 30, 2, 16, 1), 0.25, dtype=np.float32),
    )
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert records[-1]["source"] == "draft_repair_accept"


def test_risk_router_falls_back_when_repaired_action_still_rejects(tmp_path):
    clients = []

    class _RepairRejectClient:
        def __init__(self, role):
            self.role = role
            self.calls = []

        def infer(self, obs):
            self.calls.append(dict(obs))
            if obs.get("verify_action"):
                ret = {"accepted_prefix": 0, "raw_valid_prefix": 0}
                if obs.get("return_repair"):
                    ret.update({
                        "repair_action": np.full((16, 2, 16), 2.0, dtype=np.float32),
                        "repair_action_latent": np.full((1, 30, 2, 16, 1), 0.25, dtype=np.float32),
                    })
                return ret
            if obs.get("return_action_latent"):
                return {
                    "action": np.ones((16, 2, 16), dtype=np.float32),
                    "action_latent": np.zeros((1, 30, 2, 16, 1), dtype=np.float32),
                    "video_latent": np.zeros((1, 48, 2, 24, 20), dtype=np.float32),
                }
            return {"action": np.full((16, 2, 16), 3.0, dtype=np.float32)}

    def factory(host, port):
        client = _RepairRejectClient("draft" if not clients else "teacher")
        clients.append(client)
        return client

    log_path = tmp_path / "metrics.jsonl"
    policy = RiskRouterClientPolicy(
        draft_port=1,
        teacher_port=2,
        teacher_cache_mode="sync",
        risk_low=0.0,
        risk_high=0.5,
        repair_enable=True,
        log_path=str(log_path),
        client_factory=factory,
    )
