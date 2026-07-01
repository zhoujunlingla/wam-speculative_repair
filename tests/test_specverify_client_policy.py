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
    ret = policy.infer({"obs": ["x"], "state": np.zeros((16, 2, 16), dtype=np.float32)})
    assert ret["action"].shape == (16, 2, 16)
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert records[-1]["source"] == "teacher_world_verify_reject"
    assert records[-1]["verify"]["world_verify"]["world_pass"] is False


def test_risk_router_rejects_initial_partial_prefix_before_cache_update(tmp_path):
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
    assert records[-1]["source"] == "teacher_verify_reject"
    assert records[-1]["verify"]["initial_partial_prefix_rejected"] is True
