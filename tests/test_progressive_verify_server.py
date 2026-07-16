from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "wan_va"))

import wan_va_server as server_module  # noqa: E402


def _fake_server(monkeypatch):
    server = server_module.VA_Server.__new__(server_module.VA_Server)
    server.device = torch.device("cpu")
    server.dtype = torch.float32
    server.action_mask = torch.ones(30, dtype=torch.bool)
    server.action_per_frame = 16
    server.job_config = SimpleNamespace(frame_chunk_size=2)
    server.verify_scheduler = object()
    server.cache_name = "pos"
    server.forward_action_only_verify = lambda noisy, timesteps, **kwargs: torch.zeros_like(noisy)
    server.postprocess_action = lambda value: np.zeros((1, 16, 2, 16), dtype=np.float32)
    monkeypatch.setattr(
        server_module,
        "scheduler_add_noise_batched",
        lambda *, clean, **kwargs: clean.clone(),
    )
    monkeypatch.setattr(
        server_module,
        "scheduler_step_to_final_batched",
        lambda *, sample, **kwargs: sample.clone(),
    )
    return server


def test_live_progressive_k_stops_after_certified_first_probe(monkeypatch):
    server = _fake_server(monkeypatch)
    result = server.verify_action_chunk(
        torch.zeros(1, 30, 2, 16, 1),
        tau_timesteps=(50.0, 100.0),
        threshold=0.15,
        gripper_consensus=True,
        adaptive_k_live=True,
        adaptive_k_distance_threshold=0.05,
    )

    assert result["requested_verify_k"] == 2
    assert result["effective_verify_k"] == 1
    assert result["accepted_prefix"] == 32
    assert result["adaptive_k_live_accepted"] is True
    assert result["tau_timesteps"].tolist() == [50.0]


def test_live_progressive_k_runs_second_probe_when_certificate_abstains(monkeypatch):
    server = _fake_server(monkeypatch)
    result = server.verify_action_chunk(
        torch.zeros(1, 30, 2, 16, 1),
        tau_timesteps=(50.0, 100.0),
        threshold=0.15,
        gripper_consensus=True,
        adaptive_k_live=True,
        adaptive_k_distance_threshold=0.05,
        decoded_draft_gripper_switch=True,
    )

    assert result["effective_verify_k"] == 2
    assert result["adaptive_k_live_accepted"] is False
    assert result["tau_timesteps"].tolist() == [50.0, 100.0]
