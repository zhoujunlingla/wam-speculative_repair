import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wan_va"))

from specverify import (
    action_verify_frame_start,
    make_verify_scheduler,
    quantize_prefix_to_frame_boundary,
    sample_verify_noise_like,
    scheduler_add_noise_batched,
    scheduler_step_to_final_batched,
)


def test_verify_noise_seed_is_reproducible():
    clean = torch.zeros(1, 3, 2, 4, 1)
    first = sample_verify_noise_like(clean, seed=1234)
    second = sample_verify_noise_like(clean, seed=1234)
    other = sample_verify_noise_like(clean, seed=1235)

    assert torch.equal(first, second)
    assert not torch.equal(first, other)


def test_action_verify_runs_each_tau_as_a_complete_cfg_batch():
    from wan_va_server import VA_Server

    calls = []

    def prepare(_, action, _latent_t, action_t, _latent_cond, _action_cond, frame_st_id):
        frame_count, action_count = action.shape[2:4]
        return {"action_res_lst": {
            "noisy_latents": action,
            "timesteps": torch.full((frame_count,), float(action_t)),
            "grid_id": torch.zeros(frame_count * action_count, 3),
            "text_emb": torch.zeros(1, 1, 1),
        }}

    def repeat_cfg(data):
        data = dict(data)
        data["noisy_latents"] = data["noisy_latents"].repeat(2, 1, 1, 1, 1)
        data["timesteps"] = data["timesteps"][None].repeat(2, 1)
        data["grid_id"] = data["grid_id"][None].repeat(2, 1, 1)
        data["text_emb"] = torch.zeros(2, 1, 1)
        return data

    def transformer(data, **_kwargs):
        batch, channels, frames, actions, _ = data["noisy_latents"].shape
        calls.append((batch, data["timesteps"][:, 0].tolist()))
        return torch.zeros(batch, frames * actions, channels)

    fake = SimpleNamespace(
        prompt_embeds=torch.zeros(1, 1, 1),
        device=torch.device("cpu"),
        dtype=torch.float32,
        cache_name="test",
        action_mask=torch.ones(3, dtype=torch.bool),
        action_per_frame=4,
        job_config=SimpleNamespace(action_dim=3, action_guidance_scale=1.0),
        _prepare_latent_input=prepare,
        _repeat_input_for_cfg=repeat_cfg,
        transformer=transformer,
    )
    output = VA_Server.forward_action_only_verify(
        fake,
        torch.zeros(2, 3, 2, 4, 1),
        torch.tensor([150.0, 300.0]),
        frame_st_id=2,
    )

    assert output.shape == (2, 3, 2, 4, 1)
    assert calls == [(2, [150.0, 150.0]), (2, [300.0, 300.0])]


def test_verify_scheduler_keeps_full_resolution_after_teacher_scheduler_is_coarse():
    scheduler = make_verify_scheduler(action_snr_shift=1.0)

    for timestep in (150.0, 300.0):
        closest = torch.min(torch.abs(scheduler.timesteps - timestep)).item()
        assert closest < 1e-4


def test_chunk_zero_excludes_conditioned_first_action_frame():
    assert action_verify_frame_start(frame_st_id=0) == 1
    assert action_verify_frame_start(frame_st_id=2) == 0


def test_prefix_is_quantized_to_observation_frame_boundaries():
    assert quantize_prefix_to_frame_boundary(0, action_per_frame=16, frame_chunk_size=2) == 0
    assert quantize_prefix_to_frame_boundary(8, action_per_frame=16, frame_chunk_size=2) == 0
    assert quantize_prefix_to_frame_boundary(16, action_per_frame=16, frame_chunk_size=2) == 16
    assert quantize_prefix_to_frame_boundary(31, action_per_frame=16, frame_chunk_size=2) == 16
    assert quantize_prefix_to_frame_boundary(99, action_per_frame=16, frame_chunk_size=2) == 32


def test_batched_step_to_final_handles_distinct_verifier_timesteps():
    scheduler = make_verify_scheduler(action_snr_shift=1.0)
    clean = torch.randn(2, 3, 2, 4, 1)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([150.0, 300.0])
    z_tau = scheduler_add_noise_batched(
        scheduler=scheduler,
        clean=clean,
        noise=noise,
        timesteps=timesteps,
    )
    perfect_velocity = (noise - clean)

    recon = scheduler_step_to_final_batched(
        scheduler=scheduler,
        model_output=perfect_velocity,
        timesteps=timesteps,
        sample=z_tau,
    )

    assert torch.allclose(recon, clean, atol=1e-5)
