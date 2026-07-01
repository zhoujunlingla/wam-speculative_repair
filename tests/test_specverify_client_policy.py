import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.robotwin.specverify_client_policy import (
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
