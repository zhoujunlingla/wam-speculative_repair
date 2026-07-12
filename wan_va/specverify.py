# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Lightweight math helpers for Realtime-VLA-FLASH action verification."""

import math

import torch

try:
    from .utils import FlowMatchScheduler
except ImportError:  # wan_va_server.py also supports direct script execution.
    from utils import FlowMatchScheduler


CONTINUOUS_CHANNELS = tuple(range(14))
GRIPPER_CHANNELS = (28, 29)


def latent_frame_motion_stats(
    latents: torch.Tensor,
    top_fraction: float = 0.1,
) -> dict[str, float]:
    """Summarize adjacent-frame motion already present in a video latent."""

    if latents.ndim != 5 or latents.shape[2] < 2:
        raise ValueError("latents must have shape [B,C,F,H,W] with F >= 2")
    if not math.isfinite(float(top_fraction)) or not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    delta = latents[:, :, 1:] - latents[:, :, :-1]
    motion = torch.linalg.vector_norm(delta.float(), ord=2, dim=1) / math.sqrt(
        latents.shape[1]
    )
    flat = motion.flatten()
    top_count = max(1, math.ceil(flat.numel() * float(top_fraction)))
    top = torch.topk(flat, top_count).values
    eps = torch.finfo(flat.dtype).eps
    median = torch.quantile(flat, 0.5)
    return {
        "global_mean": float(flat.mean().item()),
        "median": float(median.item()),
        "top_mean": float(top.mean().item()),
        "top_relative": float((top.mean() / median.clamp_min(eps)).item()),
        "top_concentration": float((top.sum() / flat.sum().clamp_min(eps)).item()),
    }


def sample_verify_noise_like(clean: torch.Tensor, seed=None) -> torch.Tensor:
    """Sample the one Gaussian probe shared by every verifier timestep."""

    if seed is None:
        return torch.randn_like(clean)
    generator = torch.Generator(device=clean.device)
    generator.manual_seed(int(seed))
    return torch.randn(
        clean.shape,
        dtype=clean.dtype,
        device=clean.device,
        generator=generator,
    )


def make_verify_scheduler(action_snr_shift: float) -> FlowMatchScheduler:
    """Build a full-resolution scheduler independent of inference steps."""

    scheduler = FlowMatchScheduler(
        shift=action_snr_shift,
        sigma_min=0.0,
        extra_one_step=True,
    )
    scheduler.set_timesteps(1000, training=True)
    return scheduler


def action_verify_frame_start(frame_st_id: int) -> int:
    """Skip LingBot's conditioned first action frame in the initial chunk."""

    return 1 if frame_st_id == 0 else 0


def quantize_prefix_to_frame_boundary(
    prefix_len: int,
    *,
    action_per_frame: int,
    frame_chunk_size: int,
) -> int:
    """Floor a low-level action prefix to a temporal-latent boundary."""

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
    negative_prompt_embeds: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
) -> dict:
    """Pack K action/timestep rows, with optional CFG, for one forward."""

    if noisy_action.ndim != 5 or noisy_action.shape[-1] != 1:
        raise ValueError("noisy_action must have shape [K, C, F, N, 1]")
    batch_size, _, frame_count, _, _ = noisy_action.shape
    timesteps = timesteps.to(device=noisy_action.device, dtype=torch.float32).flatten()
    if timesteps.numel() != batch_size:
        raise ValueError("timesteps must contain one value per verifier row")

    def repeat_batch(value, name):
        value = value.to(device=noisy_action.device, dtype=dtype)
        if value.shape[0] == 1:
            return value.repeat(batch_size, 1, 1)
        if value.shape[0] != batch_size:
            raise ValueError(f"{name} batch must be 1 or K")
        return value

    text_emb = repeat_batch(prompt_embeds, "prompt_embeds")
    if grid_id.ndim == 2:
        grid_id = grid_id.to(noisy_action.device)[None].repeat(batch_size, 1, 1)
    elif grid_id.ndim == 3 and grid_id.shape[0] == batch_size:
        grid_id = grid_id.to(noisy_action.device)
    else:
        raise ValueError("grid_id must have shape [tokens, 3] or [K, tokens, 3]")

    timestep_rows = timesteps[:, None].repeat(1, frame_count)
    if conditioned_frame_count:
        timestep_rows[:, :conditioned_frame_count] = 0

    if float(guidance_scale) > 1:
        if negative_prompt_embeds is None:
            raise RuntimeError("action CFG verification requires negative prompt embeds")
        negative_emb = repeat_batch(negative_prompt_embeds, "negative_prompt_embeds")
        noisy_action = torch.cat([noisy_action, noisy_action], dim=0)
        text_emb = torch.cat([text_emb, negative_emb], dim=0)
        grid_id = torch.cat([grid_id, grid_id], dim=0)
        timestep_rows = torch.cat([timestep_rows, timestep_rows], dim=0)

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
    """Interpolate K clean rows with one shared probe at K sigmas."""

    timesteps = timesteps.to(device=clean.device, dtype=torch.float32).flatten()
    if clean.shape != noise.shape:
        raise ValueError("clean and noise must have identical shape")
    if clean.shape[0] != timesteps.numel():
        raise ValueError("timesteps must contain one value per verifier row")
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
    """Apply the scheduler's exact ``to_final`` reconstruction to K rows."""

    timesteps = timesteps.to(device=model_output.device, dtype=torch.float32).flatten()
    if model_output.shape != sample.shape or model_output.shape[0] != timesteps.numel():
        raise ValueError("model_output, sample, and timesteps must share batch K")
    return torch.cat([
        scheduler.step(
            model_output[row:row + 1],
            timestep.detach().cpu(),
            sample[row:row + 1],
            to_final=True,
        )
        for row, timestep in enumerate(timesteps)
    ], dim=0)


def normalized_l2_distances(
    reconstructed: torch.Tensor,
    draft: torch.Tensor,
    *,
    continuous_channels=CONTINUOUS_CHANNELS,
) -> torch.Tensor:
    """Return per-tau/per-action L2 divided by sqrt(channel count)."""

    if reconstructed.ndim != 5 or reconstructed.shape[-1] != 1:
        raise ValueError("reconstructed must have shape [K, C, F, N, 1]")
    if draft.ndim != 5 or draft.shape[-1] != 1:
        raise ValueError("draft must have shape [1|K, C, F, N, 1]")
    if draft.shape[0] not in (1, reconstructed.shape[0]) or draft.shape[1:] != reconstructed.shape[1:]:
        raise ValueError("draft must match reconstructed except for an optional singleton batch")
    channels = tuple(int(channel) for channel in continuous_channels)
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("continuous_channels must be non-empty and unique")
    if min(channels) < 0 or max(channels) >= reconstructed.shape[1]:
        raise ValueError("continuous channel index is outside the action tensor")

    diff = (reconstructed[:, channels, :, :, 0] - draft[:, channels, :, :, 0]).float()
    return torch.linalg.vector_norm(diff, ord=2, dim=1) / math.sqrt(len(channels))


def longest_prefix_min_over_k(
    distances: torch.Tensor,
    threshold: float,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Return the minimum longest prefix across K verifier rows."""

    if distances.ndim < 2 or distances.shape[0] < 1:
        raise ValueError("distances must have shape [K, ...actions]")
    if not math.isfinite(float(threshold)) or float(threshold) < 0:
        raise ValueError("threshold must be finite and non-negative")
    flat = distances.reshape(distances.shape[0], -1)
    ok = flat <= float(threshold)
    prefix_by_tau = torch.cumprod(ok.to(torch.int64), dim=1).sum(dim=1)
    raw_prefix = int(prefix_by_tau.min().item())
    return raw_prefix, prefix_by_tau, ok.all(dim=0)


def stitch_action_prefix(
    draft: torch.Tensor,
    teacher_endpoint: torch.Tensor,
    prefix_len: int,
) -> torch.Tensor:
    """Use the draft for the accepted prefix and teacher endpoint for its tail."""

    if draft.ndim != 5 or draft.shape != teacher_endpoint.shape or draft.shape[-1] != 1:
        raise ValueError("draft and teacher_endpoint must share shape [B, C, F, N, 1]")
    horizon = draft.shape[2] * draft.shape[3]
    prefix_len = max(0, min(int(prefix_len), horizon))
    prefix_mask = torch.arange(horizon, device=draft.device) < prefix_len
    prefix_mask = prefix_mask.view(1, 1, draft.shape[2], draft.shape[3], 1)
    return torch.where(prefix_mask, draft, teacher_endpoint)


def gripper_switch_info(
    action: torch.Tensor,
    *,
    max_prefix: int | None = None,
    previous=None,
    threshold: float = 0.0,
    gripper_channels=GRIPPER_CHANNELS,
) -> dict:
    """Locate the first normalized-latent gripper phase boundary."""

    if action.ndim != 5 or action.shape[0] != 1 or action.shape[-1] != 1:
        raise ValueError("action must have shape [1, C, F, N, 1]")
    channels = tuple(int(channel) for channel in gripper_channels)
    if not channels or min(channels) < 0 or max(channels) >= action.shape[1]:
        raise ValueError("gripper channel index is outside the action tensor")

    values = action[0, channels, :, :, 0].float().reshape(len(channels), -1)
    states = values >= float(threshold)
    switches = torch.zeros_like(states)
    switches[:, 1:] = states[:, 1:] != states[:, :-1]
    if previous is not None:
        previous = torch.as_tensor(previous, device=action.device).flatten()
        if previous.numel() == action.shape[1]:
            previous = previous[list(channels)]
        elif previous.numel() != len(channels):
            raise ValueError("previous must contain all action channels or one value per gripper")
        switches[:, 0] = states[:, 0] != (previous.float() >= float(threshold))

    horizon = values.shape[1]
    limit = horizon if max_prefix is None else max(0, min(int(max_prefix), horizon))
    switch_indices = []
    for row in switches[:, :limit]:
        indices = row.nonzero(as_tuple=False).flatten()
        switch_indices.append(int(indices[0].item()) if indices.numel() else None)
    present = [index for index in switch_indices if index is not None]
    first_index = min(present) if present else None
    first_channels = [
        channel for channel, index in zip(channels, switch_indices)
        if index == first_index
    ] if first_index is not None else []
    return {
        "has_switch": first_index is not None,
        "first_index": first_index,
        "channels": first_channels,
        "switch_indices": switch_indices,
    }


def gripper_consensus_prefix(
    reconstructed: torch.Tensor,
    draft: torch.Tensor,
    *,
    max_prefix: int,
    threshold: float = 0.0,
    gripper_channels=GRIPPER_CHANNELS,
) -> tuple[int, int | None]:
    """Bound a prefix at the first draft/teacher gripper phase disagreement."""

    if reconstructed.ndim != 5 or draft.ndim != 5:
        raise ValueError("actions must have shape [K, C, F, N, 1]")
    if draft.shape[0] != 1 or reconstructed.shape[1:] != draft.shape[1:]:
        raise ValueError("draft must contain one chunk matching every reconstruction")
    if reconstructed.shape[0] < 2:
        raise ValueError("cross-tau gripper consensus requires at least two probes")
    channels = tuple(int(channel) for channel in gripper_channels)
    if not channels or min(channels) < 0 or max(channels) >= draft.shape[1]:
        raise ValueError("gripper channel index is outside the action tensor")

    horizon = draft.shape[2] * draft.shape[3]
    limit = max(0, min(int(max_prefix), horizon))
    draft_phase = draft[0, channels, :, :, 0].reshape(len(channels), horizon) >= threshold
    teacher_phase = reconstructed[:, channels, :, :, 0].reshape(
        reconstructed.shape[0], len(channels), horizon
    ) >= threshold
    disagreement = torch.any(
        teacher_phase[:, :, :limit] != draft_phase[None, :, :limit], dim=(0, 1)
    )
    indices = disagreement.nonzero(as_tuple=False).flatten()
    first = int(indices[0].item()) if indices.numel() else None
    return (limit if first is None else first), first
