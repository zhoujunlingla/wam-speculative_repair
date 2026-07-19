"""Realtime-VLA-FLASH orchestration for local draft and teacher models."""

from __future__ import annotations

import json
import hashlib
import operator
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from adaptive_verify import k1_full_accept_certificate


def _stable_digest(value) -> str:
    digest = hashlib.sha256()

    def update(item) -> None:
        if isinstance(item, dict):
            digest.update(b"{")
            for key in sorted(item):
                update(str(key))
                update(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple, deque)):
            digest.update(b"[")
            for child in item:
                update(child)
            digest.update(b"]")
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        elif isinstance(item, np.generic):
            update(item.item())
        else:
            digest.update(repr(item).encode())
            digest.update(b";")

    update(value)
    return digest.hexdigest()


def quantize_prefix(prefix_len: int, action_per_frame: int, horizon: int) -> int:
    """Clamp a low-level action prefix to a real observation boundary."""

    if action_per_frame <= 0:
        raise ValueError("action_per_frame must be positive")
    prefix_len = max(0, min(int(prefix_len), int(horizon)))
    return (prefix_len // action_per_frame) * action_per_frame


def slice_action_prefix(action: np.ndarray, prefix_len: int) -> np.ndarray:
    """Slice a ``[C, F, H]`` action chunk at frame boundaries."""

    action = np.asarray(action)
    if action.ndim != 3:
        raise ValueError("action must have shape [C, F, H]")
    horizon = action.shape[1] * action.shape[2]
    prefix_len = quantize_prefix(prefix_len, action.shape[2], horizon)
    return np.ascontiguousarray(action[:, : prefix_len // action.shape[2], :])


def first_gripper_switch(
    action: np.ndarray,
    channels: Iterable[int] = (7, 15),
    threshold: float = 0.5,
    previous=None,
) -> Optional[int]:
    """Return the first low-level step entering a new gripper phase."""

    action = np.asarray(action)
    if action.ndim != 3:
        raise ValueError("action must have shape [C, F, H]")
    channels = tuple(channel for channel in channels if 0 <= channel < action.shape[0])
    if not channels:
        return None
    steps = action.transpose(1, 2, 0).reshape(-1, action.shape[0])
    if len(steps) == 0:
        return None
    phase = steps[:, channels] >= threshold
    if previous is not None:
        previous = np.asarray(previous).reshape(-1)
        if previous.size == action.shape[0]:
            previous = previous[list(channels)]
        elif previous.size != len(channels):
            raise ValueError("previous must contain every action or gripper channel")
        if np.any(phase[0] != (previous >= threshold)):
            return 0
    if len(steps) < 2:
        return None
    switches = np.flatnonzero(np.any(phase[1:] != phase[:-1], axis=1))
    return None if switches.size == 0 else int(switches[0] + 1)


def normalized_l2_step_distances(
    draft_action: np.ndarray,
    teacher_actions: np.ndarray,
    continuous_channels: Iterable[int] = range(14),
) -> np.ndarray:
    """Return ``[K, T]`` RMS distances over normalized continuous channels."""

    draft = _as_bcfh(draft_action)
    teachers = _as_bcfh(teacher_actions)
    if draft.shape[0] != 1:
        raise ValueError("draft_action must contain one action chunk")
    if draft.shape[1:] != teachers.shape[1:]:
        raise ValueError("draft and teacher action shapes must match after batch")
    channels = tuple(int(channel) for channel in continuous_channels)
    if not channels or min(channels) < 0 or max(channels) >= draft.shape[1]:
        raise ValueError("continuous_channels are invalid for the action shape")
    delta = teachers[:, channels] - draft[:, channels]
    return np.sqrt(np.mean(np.square(delta, dtype=np.float32), axis=1)).reshape(
        teachers.shape[0], -1
    )


def continuous_action_dynamics_stats(
    action_latent: np.ndarray,
    continuous_channels: Iterable[int] = range(14),
    *,
    skip_steps: int = 0,
) -> dict[str, float | int | None]:
    """Describe normalized continuous-action dynamics without routing on them."""

    action = np.asarray(action_latent, dtype=np.float32)
    if action.ndim != 5 or action.shape[0] != 1 or action.shape[-1] != 1:
        raise ValueError("action_latent must have shape [1, C, F, N, 1]")
    channels = tuple(int(channel) for channel in continuous_channels)
    if not channels or min(channels) < 0 or max(channels) >= action.shape[1]:
        raise ValueError("continuous_channels are invalid for the action latent")
    trace = action[0, channels, :, :, 0].reshape(len(channels), -1)
    skip_steps = int(skip_steps)
    if skip_steps < 0 or skip_steps >= trace.shape[1]:
        raise ValueError("skip_steps must leave at least one action step")
    trace = trace[:, skip_steps:]
    if not np.all(np.isfinite(trace)):
        raise ValueError("action_latent must be finite")

    def finite_difference_stats(order: int) -> tuple[float | None, float | None]:
        values = np.diff(trace, n=order, axis=1)
        if values.shape[1] == 0:
            return None, None
        per_step = np.sqrt(np.mean(np.square(values, dtype=np.float32), axis=0))
        return float(np.sqrt(np.mean(np.square(values, dtype=np.float32)))), float(
            np.max(per_step)
        )

    velocity_rms, velocity_max = finite_difference_stats(1)
    acceleration_rms, acceleration_max = finite_difference_stats(2)
    jerk_rms, jerk_max = finite_difference_stats(3)
    return {
        "continuous_channels": len(channels),
        "action_steps": int(trace.shape[1]),
        "skipped_conditioned_steps": skip_steps,
        "velocity_rms": velocity_rms,
        "velocity_max_step_rms": velocity_max,
        "acceleration_rms": acceleration_rms,
        "acceleration_max_step_rms": acceleration_max,
        "jerk_rms": jerk_rms,
        "jerk_max_step_rms": jerk_max,
    }


def relative_action_jerk(
    dynamics: dict,
    *,
    velocity_floor: float = 0.01,
) -> float:
    """Return scale-normalized jerk for routing already-computed draft dynamics."""

    if not np.isfinite(velocity_floor) or velocity_floor <= 0:
        raise ValueError("velocity_floor must be positive")
    velocity = dynamics.get("velocity_rms")
    jerk = dynamics.get("jerk_rms")
    if velocity is None or jerk is None:
        raise ValueError("velocity_rms and jerk_rms are required")
    velocity = float(velocity)
    jerk = float(jerk)
    if not np.isfinite(velocity) or not np.isfinite(jerk) or velocity < 0 or jerk < 0:
        raise ValueError("velocity_rms and jerk_rms must be finite and non-negative")
    return jerk / max(velocity, float(velocity_floor))


def longest_safe_prefix(
    distances: np.ndarray,
    threshold: float,
    action_per_frame: int = 16,
    horizon: Optional[int] = None,
) -> int:
    """Return the longest prefix passing every verifier timestep."""

    distances = np.asarray(distances)
    if distances.ndim < 2 or distances.shape[0] == 0:
        raise ValueError("distances must have shape [K, ...steps]")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    distances = distances.reshape(distances.shape[0], -1)
    horizon = distances.shape[1] if horizon is None else min(int(horizon), distances.shape[1])
    passing = np.all(distances[:, :horizon] <= threshold, axis=0)
    failures = np.flatnonzero(~passing)
    raw_prefix = horizon if failures.size == 0 else int(failures[0])
    return quantize_prefix(raw_prefix, action_per_frame, horizon)


def adaptive_k_pass_candidate(
    verify_response: dict,
    draft_action: np.ndarray,
    *,
    enabled: bool,
    verify_threshold: float,
    candidate_threshold: float,
    expected_tau_timesteps: Iterable[float],
    action_per_frame: int,
    gripper_channels: Iterable[int],
    gripper_threshold: float,
    previous_gripper=None,
) -> tuple[dict, Optional[dict]]:
    """Build a host-only K1 pass candidate from a completed K2 response."""

    telemetry = {
        "adaptive_k_shadow": bool(enabled),
        "adaptive_k_certificate_kind": None,
        "adaptive_k_fail_reason": None,
        "adaptive_k_certificate_matches_full": None,
        "adaptive_k_sentinel_distance_max": None,
        "adaptive_k_sentinel_full_prefix": None,
        "adaptive_k_sentinel_continuous_prefix": None,
        "adaptive_k_sentinel_gripper_prefix": None,
        "adaptive_k_sentinel_accepted_prefix": None,
        "adaptive_k_sentinel_gripper_failure_index": None,
        "adaptive_k_sentinel_phase_agreement": None,
        "adaptive_k_sentinel_gripper_switch": None,
    }
    if not enabled:
        return telemetry, None

    required_k = 2
    try:
        observed_k = []
        for key in (
            "requested_verify_k", "effective_verify_k", "primary_verify_forwards"
        ):
            raw_value = verify_response[key]
            if isinstance(raw_value, (bool, np.bool_)):
                return telemetry, None
            observed_k.append(operator.index(raw_value))
        observed_k = tuple(observed_k)
        distances = np.asarray(
            verify_response.get("distances"), dtype=np.float32
        )
        prefixes = np.asarray(
            verify_response.get("prefix_by_tau"), dtype=np.float64
        )
        agreements = np.asarray(
            verify_response.get("gripper_phase_agreement_by_tau"),
            dtype=np.float64,
        )
        observed_tau = np.asarray(
            verify_response.get("tau_timesteps"), dtype=np.float64
        )
        expected_tau = np.asarray(
            tuple(expected_tau_timesteps), dtype=np.float64
        )
        observed_threshold = float(verify_response["threshold"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return telemetry, None
    if observed_k != (required_k, required_k, required_k):
        return telemetry, None
    switch_indices = verify_response.get("gripper_switch_indices_by_tau")
    if (
        distances.ndim < 2
        or distances.shape[0] != required_k
        or prefixes.shape != (required_k,)
        or agreements.shape != (required_k,)
        or observed_tau.shape != (required_k,)
        or expected_tau.shape != (required_k,)
        or not isinstance(switch_indices, (list, tuple))
        or len(switch_indices) != required_k
        or not np.all(np.isfinite(distances))
        or not np.all(np.isfinite(prefixes))
        or not np.all(np.isfinite(agreements))
        or not np.all(np.isfinite(observed_tau))
        or not np.array_equal(observed_tau, expected_tau)
        or not np.isfinite(observed_threshold)
        or observed_threshold != float(verify_threshold)
        or verify_response.get("shared_noise") is not True
        or not np.all(prefixes == np.floor(prefixes))
    ):
        return telemetry, None

    draft_action = np.asarray(draft_action)
    if draft_action.ndim != 3:
        return telemetry, None
    horizon = int(draft_action.shape[1] * draft_action.shape[2])
    if distances[0].size != horizon:
        return telemetry, None
    first_distances = distances[0].reshape(-1)
    distance_max = float(np.max(first_distances))
    continuous_prefix = quantize_prefix(
        int(prefixes[0]), action_per_frame, horizon
    )
    phase_agreement = bool(float(agreements[0]) == 1.0)
    decoded_switch = first_gripper_switch(
        draft_action,
        channels=gripper_channels,
        threshold=gripper_threshold,
        previous=previous_gripper,
    )
    has_gripper_switch = bool(
        switch_indices[0] is not None
        or verify_response.get("draft_gripper_switch_index") is not None
        or decoded_switch is not None
    )
    full_prefix = continuous_prefix == horizon
    telemetry.update(
        adaptive_k_sentinel_distance_max=distance_max,
        adaptive_k_sentinel_full_prefix=full_prefix,
        adaptive_k_sentinel_continuous_prefix=continuous_prefix,
        adaptive_k_sentinel_accepted_prefix=continuous_prefix,
        adaptive_k_sentinel_phase_agreement=phase_agreement,
        adaptive_k_sentinel_gripper_switch=has_gripper_switch,
    )
    if not k1_full_accept_certificate(
        distance_max=distance_max,
        continuous_prefix=continuous_prefix,
        horizon=horizon,
        phase_agreement=float(agreements[0]),
        reconstructed_switch=switch_indices[0] is not None,
        draft_switch=verify_response.get("draft_gripper_switch_index") is not None,
        decoded_draft_switch=decoded_switch is not None,
        verify_threshold=verify_threshold,
        certificate_threshold=candidate_threshold,
    ):
        return telemetry, None

    telemetry["adaptive_k_certificate_kind"] = "pass"
    predicted = {
        "source": "draft_flash",
        "executed_action_hash": _stable_digest(draft_action),
        "accepted_prefix": horizon,
        "fallback_reason": None,
    }
    return telemetry, predicted


def infer_with_replan(model, request: dict) -> dict:
    """Retry one zero-prefix response with the exact same observation."""

    response = model.infer(request)
    if not isinstance(response, dict):
        raise TypeError("policy inference must return a dict")
    if not response.get("replan", False):
        return response
    response = model.infer(request)
    if not isinstance(response, dict):
        raise TypeError("policy inference must return a dict")
    if response.get("replan", False):
        raise RuntimeError("replan must be followed by a full action response")
    return response


def _as_bcfh(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action)
    if action.ndim == 5:
        if action.shape[-1] != 1:
            raise ValueError("five-dimensional actions must end in a singleton width")
        action = action[..., 0]
    elif action.ndim == 3:
        action = action[None]
    if action.ndim != 4:
        raise ValueError("action must have shape [C,F,H], [B,C,F,H], or [B,C,F,H,1]")
    return action


class RealtimeFlashPolicy:
    """First-full speculative policy with a last-full teacher reference."""

    def __init__(
        self,
        draft,
        teacher,
        *,
        pf_interval: int = 2,
        threshold: float = 0.15,
        tau_timesteps: Iterable[float] = (50.0, 100.0),
        action_per_frame: int = 16,
        gripper_channels: Iterable[int] = (7, 15),
        gripper_threshold: float = 0.5,
        teacher_gripper_fallback: bool = True,
        flow_budget_threshold: float = 0.0,
        flow_budget_burst_after: int = 0,
        flow_budget_burst_rounds: int = 0,
        flow_budget_burst_limit: int = 0,
        flow_budget_motion_ceiling: float = 0.0,
        delayed_error_threshold: float = 0.0,
        delayed_error_consecutive: int = 2,
        delayed_error_teacher_rounds: int = 2,
        video_motion_gate_threshold: float = 0.0,
        video_motion_jerk_gate_threshold: float = 0.0,
        video_motion_strict_verify_threshold: float = 0.10,
        video_motion_strict_prefix: int = 16,
        adaptive_k_shadow: bool = False,
        adaptive_k_live: bool = False,
        adaptive_k_distance_threshold: float = 0.05,
        profile_verify_latency: bool = False,
        equivalence_audit: bool = False,
        gripper_full_window: int = 1,
        gripper_consensus: bool = False,
        rng: Optional[np.random.Generator] = None,
        log_path: Optional[str] = None,
    ) -> None:
        if pf_interval < 0:
            raise ValueError("pf_interval must be non-negative")
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if action_per_frame <= 0:
            raise ValueError("action_per_frame must be positive")
        if not np.isfinite(flow_budget_threshold) or flow_budget_threshold < 0:
            raise ValueError("flow_budget_threshold must be non-negative")
        if (
            flow_budget_burst_after < 0
            or flow_budget_burst_rounds < 0
            or flow_budget_burst_limit < 0
        ):
            raise ValueError("flow budget burst values must be non-negative")
        if not np.isfinite(flow_budget_motion_ceiling) or flow_budget_motion_ceiling < 0:
            raise ValueError("flow_budget_motion_ceiling must be non-negative")
        if not np.isfinite(delayed_error_threshold) or delayed_error_threshold < 0:
            raise ValueError("delayed_error_threshold must be non-negative")
        if delayed_error_consecutive < 1 or delayed_error_teacher_rounds < 1:
            raise ValueError("delayed-error recovery values must be positive")
        if not np.isfinite(video_motion_gate_threshold) or video_motion_gate_threshold < 0:
            raise ValueError("video_motion_gate_threshold must be non-negative")
        if (
            not np.isfinite(video_motion_jerk_gate_threshold)
            or video_motion_jerk_gate_threshold < 0
        ):
            raise ValueError("video_motion_jerk_gate_threshold must be non-negative")
        if (
            not np.isfinite(video_motion_strict_verify_threshold)
            or video_motion_strict_verify_threshold < 0
        ):
            raise ValueError("video_motion_strict_verify_threshold must be non-negative")
        if video_motion_strict_prefix <= 0 or video_motion_strict_prefix % action_per_frame:
            raise ValueError(
                "video_motion_strict_prefix must be a positive action-frame multiple"
            )
        if video_motion_jerk_gate_threshold > 0 and video_motion_gate_threshold <= 0:
            raise ValueError("motion-jerk routing requires the video motion gate")
        if (
            not np.isfinite(adaptive_k_distance_threshold)
            or adaptive_k_distance_threshold < 0
        ):
            raise ValueError("adaptive K distance threshold must be non-negative")
        if adaptive_k_shadow and adaptive_k_live:
            raise ValueError("adaptive K shadow and live modes are mutually exclusive")
        if adaptive_k_live and (
            flow_budget_threshold > 0 or flow_budget_motion_ceiling > 0
        ):
            raise ValueError(
                "live adaptive K cannot drive a flow budget without K=2 distances"
            )
        if gripper_full_window < 1:
            raise ValueError("gripper_full_window must be positive")
        tau_timesteps = tuple(float(timestep) for timestep in tau_timesteps)
        if not tau_timesteps:
            raise ValueError("tau_timesteps must be non-empty")
        if gripper_consensus and len(tau_timesteps) < 2:
            raise ValueError("gripper consensus requires at least two tau probes")

        self.draft = draft
        self.teacher = teacher
        self.pf_interval = int(pf_interval)
        self.threshold = float(threshold)
        self.tau_timesteps = tau_timesteps
        self.action_per_frame = int(action_per_frame)
        self.gripper_channels = tuple(int(channel) for channel in gripper_channels)
        self.gripper_threshold = float(gripper_threshold)
        self.teacher_gripper_fallback = bool(teacher_gripper_fallback)
        self.flow_budget_threshold = float(flow_budget_threshold)
        self.flow_budget_burst_after = int(flow_budget_burst_after)
        self.flow_budget_burst_rounds = int(flow_budget_burst_rounds)
        self.flow_budget_burst_limit = int(flow_budget_burst_limit)
        self.flow_budget_motion_ceiling = float(flow_budget_motion_ceiling)
        self.delayed_error_threshold = float(delayed_error_threshold)
        self.delayed_error_consecutive = int(delayed_error_consecutive)
        self.delayed_error_teacher_rounds = int(delayed_error_teacher_rounds)
        self.video_motion_gate_threshold = float(video_motion_gate_threshold)
        self.video_motion_jerk_gate_threshold = float(
            video_motion_jerk_gate_threshold
        )
        self.video_motion_strict_verify_threshold = float(
            video_motion_strict_verify_threshold
        )
        self.video_motion_strict_prefix = int(video_motion_strict_prefix)
        self.adaptive_k_shadow = bool(adaptive_k_shadow)
        self.adaptive_k_live = bool(adaptive_k_live)
        self.adaptive_k_distance_threshold = float(
            adaptive_k_distance_threshold
        )
        self.profile_verify_latency = bool(profile_verify_latency)
        self.equivalence_audit = bool(equivalence_audit)
        self.gripper_full_window = int(gripper_full_window)
        self.gripper_consensus = bool(gripper_consensus)
        self.rng = rng or np.random.default_rng()
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._reset_state()

    def _reset_state(self) -> None:
        self.round_id = 0
        self.frame_st_id = 0
        self.flash_rounds_since_full = 0
        self.teacher_anchor_frame_st_id = None
        self.force_full_reason = None
        self.pending_cache_source = None
        self.pending_gripper = None
        self.pending_teacher_cache_updates = deque()
        self.last_source = None
        self.last_gripper = None
        self.flow_error_budget = 0.0
        self.flow_budget_refresh_count = 0
        self.flow_budget_bursts_used = 0
        self.teacher_burst_rounds_left = 0
        self.delayed_error_streak = 0
        self.delayed_error_trigger_count = 0
        self.delayed_error_teacher_rounds_left = 0
        self.gripper_full_rounds_left = 0
        self._draft_primed = False
        self._draft_cache_hash = _stable_digest("empty-draft-cache")
        self._teacher_cache_hash = _stable_digest("empty-teacher-cache")

    def _call(self, model, request: dict) -> dict:
        response = model.infer(request)
        if not isinstance(response, dict):
            raise TypeError("model inference must return a dict")
        return response

    def _log(self, **record) -> None:
        if not self.log_path:
            return
        record.setdefault("round_id", self.round_id)
        record.setdefault("frame_st_id", self.frame_st_id)
        if self.equivalence_audit:
            record.setdefault("cache_hash", self._cache_fingerprint())
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=True) + "\n")

    @staticmethod
    def _advance_cache_hash(current: str, request: dict) -> str:
        return _stable_digest((current, request))

    def _cache_fingerprint(self) -> str:
        pending = [_stable_digest(request)
                   for request in self.pending_teacher_cache_updates]
        return _stable_digest({
            "draft": self._draft_cache_hash,
            "teacher": self._teacher_cache_hash,
            "draft_state": (
                self.draft.cache_fingerprint()
                if hasattr(self.draft, "cache_fingerprint") else None
            ),
            "teacher_state": (
                self.teacher.cache_fingerprint()
                if hasattr(self.teacher, "cache_fingerprint") else None
            ),
            "pending": pending,
            "frame_st_id": self.frame_st_id,
            "teacher_anchor_frame_st_id": self.teacher_anchor_frame_st_id,
            "pending_cache_source": self.pending_cache_source,
        })

    @staticmethod
    def _cache_frame_count(request: dict) -> int:
        state = request.get("state")
        if state is None:
            raise ValueError("cache update requires state")
        state = np.asarray(state)
        if state.ndim != 3 or state.shape[1] <= 0:
            raise ValueError("cache update state must have shape [C, F, H]")
        return int(state.shape[1])

    @staticmethod
    def _action(response: dict) -> np.ndarray:
        if "action" not in response:
            raise RuntimeError("action inference did not return action")
        action = np.asarray(response["action"])
        if action.ndim != 3:
            raise ValueError("model action must have shape [C, F, H]")
        return action

    def _stage_gripper(self, action: np.ndarray) -> None:
        action = np.asarray(action)
        channels = tuple(
            channel for channel in self.gripper_channels
            if 0 <= channel < action.shape[0]
        )
        if not channels:
            self.pending_gripper = None
            return
        steps = action.transpose(1, 2, 0).reshape(-1, action.shape[0])
        if len(steps) == 0:
            raise ValueError("executed action prefix must be non-empty")
        self.pending_gripper = steps[-1, channels].astype(np.float32, copy=True)

    def _full_reason(self) -> Optional[str]:
        if self.teacher_anchor_frame_st_id is None:
            return "initial"
        if self.force_full_reason:
            return self.force_full_reason
        if self.delayed_error_teacher_rounds_left > 0:
            return "delayed_video_error_burst"
        if self.teacher_burst_rounds_left > 0:
            return "flow_budget_burst"
        if self.gripper_full_rounds_left > 0:
            return "gripper_phase_burst"
        if self.pf_interval > 0 and self.flash_rounds_since_full >= self.pf_interval:
            return "periodic"
        return None

    def _replay_teacher_updates(self) -> int:
        replayed = 0
        while self.pending_teacher_cache_updates:
            request = self.pending_teacher_cache_updates[0]
            self._call(self.teacher, request)
            self._teacher_cache_hash = self._advance_cache_hash(
                self._teacher_cache_hash, request
            )
            self.pending_teacher_cache_updates.popleft()
            replayed += 1
        return replayed

    def _reset(self, request: dict) -> dict:
        start = time.perf_counter()
        self._reset_state()
        paired_rng_seed = request.get("paired_rng_seed")
        if paired_rng_seed is not None:
            seed = int(paired_rng_seed)
            if seed < 0:
                raise ValueError("paired_rng_seed must be non-negative")
            self.rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        self._call(self.draft, request)
        response = self._call(self.teacher, request)
        self._draft_cache_hash = self._advance_cache_hash(
            self._draft_cache_hash, request
        )
        self._teacher_cache_hash = self._advance_cache_hash(
            self._teacher_cache_hash, request
        )
        self._log(
            source="reset",
            episode_index=request.get("episode_index"),
            scene_seed=request.get("scene_seed"),
            paired_rng_seed=paired_rng_seed,
            prompt=request.get("prompt"),
            elapsed_sec=time.perf_counter() - start,
        )
        return response

    def _update_cache(self, request: dict) -> dict:
        start = time.perf_counter()
        source = self.pending_cache_source
        if source not in ("full", "flash"):
            raise RuntimeError("cache update received without an executed action")
        frame_count = self._cache_frame_count(request)

        draft_request = dict(request)
        draft_request["compare_video_prediction"] = source == "flash"
        draft_response = self._call(self.draft, draft_request)
        self._draft_cache_hash = self._advance_cache_hash(
            self._draft_cache_hash, draft_request
        )
        if source == "full":
            self._call(self.teacher, dict(request))
            self._teacher_cache_hash = self._advance_cache_hash(
                self._teacher_cache_hash, request
            )
        else:
            self.pending_teacher_cache_updates.append(dict(request))

        self.frame_st_id += frame_count
        if source == "full":
            self.teacher_anchor_frame_st_id = self.frame_st_id
        self.last_gripper = self.pending_gripper
        self.pending_gripper = None
        self.pending_cache_source = None
        delayed_video_error = draft_response.get("delayed_video_error")
        delayed_error_triggered = False
        if source == "full":
            self.delayed_error_streak = 0
        elif self.delayed_error_threshold > 0 and delayed_video_error is not None:
            latent_nrmse = float(delayed_video_error["latent_nrmse"])
            if not np.isfinite(latent_nrmse):
                raise ValueError("delayed video latent_nrmse must be finite")
            self.delayed_error_streak = (
                self.delayed_error_streak + 1
                if latent_nrmse > self.delayed_error_threshold
                else 0
            )
            if self.delayed_error_streak >= self.delayed_error_consecutive:
                delayed_error_triggered = True
                self.delayed_error_trigger_count += 1
                self.delayed_error_teacher_rounds_left = max(
                    self.delayed_error_teacher_rounds_left,
                    self.delayed_error_teacher_rounds,
                )
                if self.force_full_reason is None:
                    self.force_full_reason = "delayed_video_error"

        self._log(
            source=f"{source}_cache_update",
            cache_frame_count=frame_count,
            pending_teacher_cache_updates=len(self.pending_teacher_cache_updates),
            delayed_video_error=delayed_video_error,
            delayed_error_streak=self.delayed_error_streak,
            delayed_error_triggered=delayed_error_triggered,
            delayed_error_trigger_count=self.delayed_error_trigger_count,
            delayed_error_teacher_rounds_left=self.delayed_error_teacher_rounds_left,
            elapsed_sec=time.perf_counter() - start,
        )
        return {}

    def _full(self, request: dict, reason: str) -> dict:
        start = time.perf_counter()
        flow_error_budget = self.flow_error_budget
        replayed = self._replay_teacher_updates()
        if not self._draft_primed:
            prime_request = dict(request)
            prime_request["prime_only"] = True
            self._call(self.draft, prime_request)
            self._draft_cache_hash = self._advance_cache_hash(
                self._draft_cache_hash, prime_request
            )
            self._draft_primed = True

        teacher_response = self._call(self.teacher, dict(request))
        action = self._action(teacher_response)
        horizon = action.shape[1] * action.shape[2]
        self.pending_cache_source = "full"
        self._stage_gripper(action)
        self.last_source = "teacher_full"
        self.flash_rounds_since_full = 0
        self.flow_error_budget = 0.0
        self.delayed_error_streak = 0
        self.force_full_reason = None
        if self.delayed_error_teacher_rounds_left > 0:
            self.delayed_error_teacher_rounds_left -= 1
        if self.teacher_burst_rounds_left > 0:
            self.teacher_burst_rounds_left -= 1
        if self.gripper_full_rounds_left > 0:
            self.gripper_full_rounds_left -= 1
        self.round_id += 1

        response = dict(teacher_response)
        response.update(
            replan=False,
            action_source=self.last_source,
            full_reason=reason,
            accepted_prefix=horizon,
        )
        self._log(
            source=self.last_source,
            full_reason=reason,
            fallback_reason=reason,
            accepted_prefix=horizon,
            executed_action_hash=(
                _stable_digest(action) if self.equivalence_audit else None
            ),
            replayed_teacher_cache_updates=replayed,
            flow_error_budget_before_reset=flow_error_budget,
            elapsed_sec=time.perf_counter() - start,
        )
        return response

    @staticmethod
    def _flow_budget_charge(distances, accepted_prefix: int) -> float:
        if distances is None or accepted_prefix <= 0:
            return 0.0
        distances = np.asarray(distances, dtype=np.float32)
        if distances.ndim < 2 or distances.shape[0] == 0:
            raise ValueError("verifier distances must have shape [K, ...steps]")
        per_step = np.max(distances.reshape(distances.shape[0], -1), axis=0)
        executed = per_step[: min(int(accepted_prefix), per_step.size)]
        return float(np.mean(executed)) if executed.size else 0.0

    def _accepted_prefix(
        self,
        action_latent: np.ndarray,
        verify_response: dict,
        horizon: int,
    ) -> int:
        if not self.teacher_gripper_fallback and \
                "accepted_prefix_before_gripper" in verify_response:
            raw_prefix = int(verify_response["accepted_prefix_before_gripper"])
        elif "accepted_prefix" in verify_response:
            raw_prefix = int(verify_response["accepted_prefix"])
        elif "raw_valid_prefix" in verify_response:
            raw_prefix = int(verify_response["raw_valid_prefix"])
        elif "distances" in verify_response:
            return longest_safe_prefix(
                verify_response["distances"],
                self.threshold,
                self.action_per_frame,
                horizon,
            )
        elif "action_reconstructions" in verify_response:
            distances = normalized_l2_step_distances(
                action_latent, verify_response["action_reconstructions"]
            )
            return longest_safe_prefix(
                distances, self.threshold, self.action_per_frame, horizon
            )
        else:
            raise RuntimeError("teacher verification did not return a prefix or distances")
        return quantize_prefix(raw_prefix, self.action_per_frame, horizon)

    def _flash(self, request: dict) -> dict:
        start = time.perf_counter()
        draft_request = dict(request)
        draft_request["return_action_latent"] = True
        draft_request["return_video_motion_stats"] = True
        draft_request["track_video_prediction"] = True
        draft_response = self._call(self.draft, draft_request)
        self._draft_primed = True
        action = self._action(draft_response)
        action_latent = draft_response.get("action_latent")
        if action_latent is None:
            raise RuntimeError("draft inference did not return action_latent")
        action_latent = np.asarray(action_latent)
        try:
            action_dynamics_stats = continuous_action_dynamics_stats(
                action_latent,
                skip_steps=self.action_per_frame if self.frame_st_id == 0 else 0,
            )
        except Exception as error:  # telemetry must not alter routing
            action_dynamics_stats = {
                "available": False,
                "reason": f"telemetry_error:{type(error).__name__}",
            }
        video_motion_stats = draft_response.get("video_motion_stats")
        horizon = action.shape[1] * action.shape[2]

        global_motion = None
        if self.video_motion_gate_threshold > 0 or self.flow_budget_motion_ceiling > 0:
            if video_motion_stats is None or "global_mean" not in video_motion_stats:
                raise RuntimeError("motion-aware routing requires draft global_mean")
            global_motion = float(video_motion_stats["global_mean"])
            if not np.isfinite(global_motion):
                raise ValueError("draft global video motion must be finite")
        high_motion = bool(
            self.video_motion_gate_threshold > 0
            and global_motion >= self.video_motion_gate_threshold
        )
        motion_jerk_ratio = None
        smooth_high_motion = False
        if high_motion:
            if self.video_motion_jerk_gate_threshold > 0:
                try:
                    motion_jerk_ratio = relative_action_jerk(action_dynamics_stats)
                except ValueError:
                    motion_jerk_ratio = float("inf")
                smooth_high_motion = (
                    motion_jerk_ratio < self.video_motion_jerk_gate_threshold
                )
            if not smooth_high_motion:
                self.force_full_reason = "video_motion_risk"
                self.last_source = "replan"
                self.round_id += 1
                response = {
                    "replan": True,
                    "action_source": self.last_source,
                    "fallback_reason": self.force_full_reason,
                    "accepted_prefix": 0,
                    "verified_prefix": None,
                }
                self._log(
                    source=self.last_source,
                    fallback_reason=self.force_full_reason,
                    accepted_prefix=0,
                    verified_prefix=None,
                    executed_action_hash=None,
                    video_motion_stats=video_motion_stats,
                    action_dynamics_stats=action_dynamics_stats,
                    motion_jerk_ratio=motion_jerk_ratio,
                    motion_jerk_route="teacher",
                    video_motion_gate_threshold=self.video_motion_gate_threshold,
                    elapsed_sec=time.perf_counter() - start,
                )
                return response

        switch_step = first_gripper_switch(
            action,
            self.gripper_channels,
            self.gripper_threshold,
            previous=self.last_gripper,
        )
        verify_request = dict(request)
        for key in (
            "adaptive_k_shadow",
            "adaptive_k_live",
            "adaptive_k_distance_threshold",
            "decoded_draft_gripper_switch",
            "profile_verify_latency",
        ):
            verify_request.pop(key, None)
        verify_request.update(
            verify_action=True,
            action_latent=action_latent,
            verify_noise=self.rng.standard_normal(action_latent.shape).astype(np.float32),
            threshold=(
                self.video_motion_strict_verify_threshold
                if smooth_high_motion
                else self.threshold
            ),
            tau_timesteps=self.tau_timesteps,
            frame_st_id=self.frame_st_id,
            gripper_consensus=self.gripper_consensus,
        )
        if self.adaptive_k_live:
            verify_request.update(
                adaptive_k_live=True,
                adaptive_k_distance_threshold=self.adaptive_k_distance_threshold,
                decoded_draft_gripper_switch=switch_step is not None,
            )
        if self.profile_verify_latency:
            verify_request["profile_verify_latency"] = True
        if self.last_gripper is not None:
            previous_phase = self.last_gripper >= self.gripper_threshold
            verify_request["previous_gripper"] = np.where(
                previous_phase, 1.0, -1.0
            ).astype(np.float32)
        verify_response = self._call(self.teacher, verify_request)
        verified_prefix = self._accepted_prefix(action_latent, verify_response, horizon)
        if smooth_high_motion and verified_prefix < horizon:
            verified_prefix = 0
        adaptive_telemetry, adaptive_prediction = adaptive_k_pass_candidate(
            verify_response,
            action,
            enabled=self.adaptive_k_shadow,
            verify_threshold=self.threshold,
            candidate_threshold=self.adaptive_k_distance_threshold,
            expected_tau_timesteps=self.tau_timesteps,
            action_per_frame=self.action_per_frame,
            gripper_channels=self.gripper_channels,
            gripper_threshold=self.gripper_threshold,
            previous_gripper=self.last_gripper,
        )
        verify_distances = verify_response.get("distances")
        if verify_distances is not None:
            verify_distances = np.asarray(verify_distances).tolist()
        verify_telemetry = {
            "tau_timesteps": verify_response.get("tau_timesteps"),
            "prefix_by_tau": verify_response.get("prefix_by_tau"),
            "verify_distances": verify_distances,
            "gripper_switch_indices_by_tau": verify_response.get(
                "gripper_switch_indices_by_tau"
            ),
            "gripper_phase_agreement_by_tau": verify_response.get(
                "gripper_phase_agreement_by_tau"
            ),
            "draft_gripper_switch_index": verify_response.get(
                "draft_gripper_switch_index"
            ),
            "video_motion_stats": video_motion_stats,
            "action_dynamics_stats": action_dynamics_stats,
            "motion_jerk_ratio": motion_jerk_ratio,
            "motion_jerk_route": "strict_k2" if smooth_high_motion else "normal_k2",
            "motion_verify_threshold": (
                self.video_motion_strict_verify_threshold
                if smooth_high_motion
                else self.threshold
            ),
            "world_flow_evidence": verify_response.get(
                "world_flow_evidence"
            ),
            "gripper_consensus_prefix": verify_response.get(
                "gripper_consensus_prefix"
            ),
            "gripper_consensus_failure_index": verify_response.get(
                "gripper_consensus_failure_index"
            ),
            "requested_verify_k": verify_response.get("requested_verify_k"),
            "effective_verify_k": verify_response.get("effective_verify_k"),
            "primary_verify_forwards": verify_response.get(
                "primary_verify_forwards"
            ),
            "adaptive_k_live": verify_response.get("adaptive_k_live", False),
            "adaptive_k_live_accepted": verify_response.get(
                "adaptive_k_live_accepted", False
            ),
            "verify_probe_latency_sec": verify_response.get(
                "verify_probe_latency_sec"
            ),
            "verify_finalize_latency_sec": verify_response.get(
                "verify_finalize_latency_sec"
            ),
            "verify_total_latency_sec": verify_response.get(
                "verify_total_latency_sec"
            ),
            **adaptive_telemetry,
        }

        teacher_gripper_switch = bool(
            verify_response.get("gripper_force_teacher", False)
        )
        gripper_consensus_failure = verify_response.get(
            "gripper_consensus_failure_index"
        )
        if (
            teacher_gripper_switch
            and self.teacher_gripper_fallback
            and not self.gripper_consensus
        ):
            verified_prefix = 0
            self.force_full_reason = "teacher_gripper_switch"
            self.gripper_full_rounds_left = self.gripper_full_window

        accepted_prefix = verified_prefix
        if (
            switch_step is not None
            and verified_prefix > 0
            and not self.gripper_consensus
        ):
            accepted_prefix = min(
                accepted_prefix,
                quantize_prefix(switch_step, self.action_per_frame, horizon),
            )
            self.force_full_reason = "gripper_switch"
            self.gripper_full_rounds_left = self.gripper_full_window
        if self.gripper_consensus and gripper_consensus_failure is not None:
            self.force_full_reason = "gripper_consensus"
            self.gripper_full_rounds_left = self.gripper_full_window

        if smooth_high_motion:
            if gripper_consensus_failure is not None:
                accepted_prefix = 0
                self.force_full_reason = "video_motion_strict_gripper"
            elif verified_prefix == horizon:
                accepted_prefix = min(
                    accepted_prefix,
                    self.video_motion_strict_prefix,
                )

        self.round_id += 1
        if accepted_prefix == 0:
            reason = self.force_full_reason or "zero_prefix"
            self.force_full_reason = reason
            self.last_source = "replan"
            if adaptive_prediction is not None:
                verify_telemetry["adaptive_k_certificate_matches_full"] = bool(
                    adaptive_prediction == {
                        "source": self.last_source,
                        "executed_action_hash": None,
                        "accepted_prefix": 0,
                        "fallback_reason": reason,
                    }
                )
            response = {
                "replan": True,
                "action_source": self.last_source,
                "fallback_reason": reason,
                "accepted_prefix": 0,
                "verified_prefix": verified_prefix,
            }
            self._log(
                source=self.last_source,
                fallback_reason=reason,
                accepted_prefix=0,
                verified_prefix=verified_prefix,
                executed_action_hash=None,
                teacher_gripper_switch=teacher_gripper_switch,
                **verify_telemetry,
                elapsed_sec=time.perf_counter() - start,
            )
            return response

        self.pending_cache_source = "flash"
        self.last_source = "draft_flash"
        self.flash_rounds_since_full += 1
        flow_budget_charge_raw = self._flow_budget_charge(
            verify_response.get("distances"), accepted_prefix
        )
        flow_budget_charge = flow_budget_charge_raw
        flow_budget_motion_reset = False
        if self.flow_budget_motion_ceiling > 0:
            if global_motion > self.flow_budget_motion_ceiling:
                self.flow_error_budget = 0.0
                flow_budget_charge = 0.0
                flow_budget_motion_reset = True
        self.flow_error_budget += flow_budget_charge
        if (
            self.force_full_reason is None
            and self.flow_budget_threshold > 0
            and self.flow_error_budget >= self.flow_budget_threshold
        ):
            self.force_full_reason = "flow_budget"
            self.flow_budget_refresh_count += 1
            if (
                self.flow_budget_burst_after > 0
                and self.flow_budget_refresh_count >= self.flow_budget_burst_after
                and (
                    self.flow_budget_burst_limit == 0
                    or self.flow_budget_bursts_used < self.flow_budget_burst_limit
                )
            ):
                self.teacher_burst_rounds_left = self.flow_budget_burst_rounds
                self.flow_budget_bursts_used += 1
        executed_action = slice_action_prefix(action, accepted_prefix)
        executed_action_hash = (
            _stable_digest(executed_action) if self.equivalence_audit else None
        )
        self._stage_gripper(executed_action)
        response = {
            "action": executed_action,
            "replan": False,
            "action_source": self.last_source,
            "accepted_prefix": accepted_prefix,
            "verified_prefix": verified_prefix,
        }
        if gripper_consensus_failure is not None:
            response.update(
                fallback_reason="gripper_consensus",
                gripper_consensus_failure_index=gripper_consensus_failure,
            )
        elif switch_step is not None and not self.gripper_consensus:
            response.update(
                fallback_reason="gripper_switch",
                gripper_switch_step=switch_step,
            )
        if adaptive_prediction is not None:
            verify_telemetry["adaptive_k_certificate_matches_full"] = bool(
                adaptive_prediction == {
                    "source": self.last_source,
                    "executed_action_hash": _stable_digest(executed_action),
                    "accepted_prefix": accepted_prefix,
                    "fallback_reason": response.get("fallback_reason"),
                }
            )
        self._log(
            source=self.last_source,
            fallback_reason=response.get("fallback_reason"),
            accepted_prefix=accepted_prefix,
            verified_prefix=verified_prefix,
            executed_action_hash=executed_action_hash,
            teacher_gripper_switch=teacher_gripper_switch,
            decoded_gripper_switch_step=switch_step,
            flow_budget_charge=flow_budget_charge,
            flow_budget_charge_raw=flow_budget_charge_raw,
            flow_budget_motion_ceiling=self.flow_budget_motion_ceiling,
            flow_budget_motion_reset=flow_budget_motion_reset,
            flow_error_budget=self.flow_error_budget,
            flow_budget_refresh_count=self.flow_budget_refresh_count,
            teacher_burst_rounds_left=self.teacher_burst_rounds_left,
            delayed_error_streak=self.delayed_error_streak,
            delayed_error_trigger_count=self.delayed_error_trigger_count,
            delayed_error_teacher_rounds_left=self.delayed_error_teacher_rounds_left,
            gripper_full_rounds_left=self.gripper_full_rounds_left,
            **verify_telemetry,
            elapsed_sec=time.perf_counter() - start,
        )
        return response

    def infer(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise TypeError("policy request must be a dict")
        if request.get("reset", False):
            return self._reset(request)
        if request.get("compute_kv_cache", False):
            return self._update_cache(request)
        if self.pending_cache_source is not None:
            raise RuntimeError("an action cache update is required before the next action")
        reason = self._full_reason()
        return self._full(request, reason) if reason else self._flash(request)
