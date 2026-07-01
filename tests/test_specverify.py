import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wan_va"))

from specverify import (
    action_verify_frame_start,
    build_verify_action_input,
    make_verify_scheduler,
    quantize_prefix_to_frame_boundary,
    scheduler_add_noise_batched,
    scheduler_step_to_final_batched,
)


def test_verify_scheduler_keeps_full_resolution_after_teacher_scheduler_is_coarse():
    scheduler = make_verify_scheduler(action_snr_shift=1.0)

    for timestep in (150.0, 300.0):
        closest = torch.min(torch.abs(scheduler.timesteps - timestep)).item()
        assert closest < 1e-4


def test_build_verify_action_input_batches_k_without_cfg_repeat():
    noisy_action = torch.zeros(2, 30, 2, 16, 1)
    prompt_embeds = torch.randn(1, 8, 4)
    grid_id = torch.arange(2 * 16 * 3).view(2 * 16, 3)
    timesteps = torch.tensor([150.0, 300.0])

    packed = build_verify_action_input(
        noisy_action=noisy_action,
        prompt_embeds=prompt_embeds,
        grid_id=grid_id,
        timesteps=timesteps,
        dtype=torch.float32,
    )

    assert packed["noisy_latents"].shape[0] == 2
    assert packed["text_emb"].shape[0] == 2
    assert packed["grid_id"].shape[0] == 2
    assert packed["timesteps"].shape == (2, 2)
    assert torch.equal(packed["timesteps"][0], torch.tensor([150.0, 150.0]))
    assert torch.equal(packed["timesteps"][1], torch.tensor([300.0, 300.0]))


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
