# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Utilities for Realtime-VLA-style action verification in LingBot-VA.

This module is intentionally lightweight: it can be imported by offline tests
without loading transformer, VAE, or flash-attention dependencies.
"""

import torch

from utils import FlowMatchScheduler


def make_verify_scheduler(action_snr_shift: float) -> FlowMatchScheduler:
    """Build an action verifier scheduler that is independent of inference.

    The normal action scheduler is mutated by ``set_timesteps(action_steps)``
    during inference. Verification timesteps such as 150 and 300 need the full
    1000-step table so the scheduler does not silently snap to a coarse
    inference grid.
    """

    scheduler = FlowMatchScheduler(
        shift=action_snr_shift,
        sigma_min=0.0,
        extra_one_step=True,
    )
    scheduler.set_timesteps(1000, training=True)
    return scheduler


def action_verify_frame_start(frame_st_id: int) -> int:
    """Return the first action-frame index that should be verified."""

    return 1 if frame_st_id == 0 else 0


def quantize_prefix_to_frame_boundary(
    prefix_len: int,
    *,
    action_per_frame: int,
    frame_chunk_size: int,
) -> int:
    """Clamp a speculative prefix to observation-frame boundaries."""

    if action_per_frame <= 0:
        raise ValueError("action_per_frame must be positive")
    if frame_chunk_size <= 0:
        raise ValueError("frame_chunk_size must be positive")

    max_len = action_per_frame * frame_chunk_size
    prefix_len = max(0, min(int(prefix_len), max_len))
    return (prefix_len // action_per_frame) * action_per_frame


def build_verify_action_input(
    *,
    noisy_action: torch.Tensor,
    prompt_embeds: torch.Tensor,
    grid_id: torch.Tensor,
    timesteps: torch.Tensor,
    dtype: torch.dtype,
    conditioned_frame_count: int = 0,
) -> dict:
    """Pack a K-batch action verifier input without classifier-free guidance.

    ``wan_va_server._repeat_input_for_cfg`` assumes batch size 1 and is unsafe
    for K parallel verification timesteps. This helper repeats only the prompt,
    grid, and per-row timestep metadata needed by the action branch.
    """

    if noisy_action.ndim != 5:
        raise ValueError("noisy_action must have shape [K, C, F, N, 1]")
    batch_size = noisy_action.shape[0]
    frame_count = noisy_action.shape[2]
    timesteps = timesteps.to(device=noisy_action.device, dtype=torch.float32).flatten()
    if timesteps.numel() != batch_size:
        raise ValueError("timesteps must contain one value per verifier batch")

    text_emb = prompt_embeds.to(device=noisy_action.device, dtype=dtype)
    if text_emb.shape[0] == 1:
        text_emb = text_emb.repeat(batch_size, 1, 1)
    elif text_emb.shape[0] != batch_size:
        raise ValueError("prompt_embeds batch must be 1 or match noisy_action batch")

    if grid_id.ndim == 2:
        grid_id = grid_id.to(noisy_action.device)[None].repeat(batch_size, 1, 1)
    elif grid_id.ndim == 3 and grid_id.shape[0] == batch_size:
        grid_id = grid_id.to(noisy_action.device)
    else:
        raise ValueError("grid_id must have shape [tokens, 3] or [K, tokens, 3]")

    timestep_rows = timesteps[:, None].repeat(1, frame_count)
    if conditioned_frame_count:
        timestep_rows[:, :conditioned_frame_count] = 0

    return {
        "noisy_latents": noisy_action,
        "text_emb": text_emb,
        "grid_id": grid_id,
        "timesteps": timestep_rows,
    }


def scheduler_add_noise_batched(
    *,
    scheduler: FlowMatchScheduler,
    clean: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Interpolate clean action and noise with one tau per verifier batch."""

    timesteps = timesteps.to(device=clean.device, dtype=torch.float32).flatten()
    if clean.shape != noise.shape:
        raise ValueError("clean and noise must have identical shape")
    if clean.shape[0] != timesteps.numel():
        raise ValueError("timesteps must contain one value per verifier batch")

    timestep_ids = torch.argmin(
        (scheduler.timesteps[:, None].to(timesteps.device) - timesteps[None]).abs(),
        dim=0,
    )
    sigma = scheduler.sigmas.to(device=clean.device, dtype=clean.dtype)[timestep_ids]
    sigma = sigma.view(clean.shape[0], *([1] * (clean.ndim - 1)))
    return (1 - sigma) * clean + sigma * noise


def scheduler_step_to_final_batched(
    *,
    scheduler: FlowMatchScheduler,
    model_output: torch.Tensor,
    timesteps: torch.Tensor,
    sample: torch.Tensor,
) -> torch.Tensor:
    """Run ``scheduler.step(..., to_final=True)`` for K verifier rows.

    ``FlowMatchScheduler.step`` accepts one timestep at a time, while
    verification evaluates multiple tau levels in parallel. The model forward
    can be batched; the scheduler update is kept explicit and exact here.
    """

    timesteps = timesteps.to(device=model_output.device, dtype=torch.float32).flatten()
    if model_output.shape[0] != timesteps.numel() or sample.shape[0] != timesteps.numel():
        raise ValueError("model_output, sample, and timesteps must share batch K")

    outputs = []
    for row, timestep in enumerate(timesteps):
        outputs.append(
            scheduler.step(
                model_output[row:row + 1],
                timestep.detach().cpu(),
                sample[row:row + 1],
                to_final=True,
            )
        )
    return torch.cat(outputs, dim=0)


def scheduler_step_to_timestep_batched(
    *,
    scheduler: FlowMatchScheduler,
    model_output: torch.Tensor,
    from_timesteps: torch.Tensor,
    to_timesteps: torch.Tensor,
    sample: torch.Tensor,
) -> torch.Tensor:
    """Step batched flow samples from tau to another tau."""

    from_timesteps = from_timesteps.to(device=model_output.device, dtype=torch.float32).flatten()
    to_timesteps = to_timesteps.to(device=model_output.device, dtype=torch.float32).flatten()
    if model_output.shape[0] != from_timesteps.numel() or sample.shape[0] != from_timesteps.numel():
        raise ValueError("model_output, sample, and from_timesteps must share batch K")
    if to_timesteps.numel() == 1:
        to_timesteps = to_timesteps.repeat(from_timesteps.numel())
    if to_timesteps.numel() != from_timesteps.numel():
        raise ValueError("to_timesteps must be scalar or share batch K")

    scheduler_timesteps = scheduler.timesteps.to(from_timesteps.device)
    scheduler_sigmas = scheduler.sigmas.to(device=model_output.device, dtype=model_output.dtype)
    outputs = []
    for row, (t_from, t_to) in enumerate(zip(from_timesteps, to_timesteps)):
        from_id = torch.argmin((scheduler_timesteps - t_from).abs())
        to_id = torch.argmin((scheduler_timesteps - t_to).abs())
        sigma_from = scheduler_sigmas[from_id]
        sigma_to = scheduler_sigmas[to_id]
        outputs.append(sample[row:row + 1] + model_output[row:row + 1] * (sigma_to - sigma_from))
    return torch.cat(outputs, dim=0)
