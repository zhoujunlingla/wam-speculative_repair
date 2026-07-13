"""Realtime-VLA-FLASH orchestration for local draft and teacher models."""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


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
        delayed_error_threshold: float = 0.0,
        delayed_error_consecutive: int = 2,
        delayed_error_teacher_rounds: int = 2,
        video_motion_gate_threshold: float = 0.0,
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
        if not np.isfinite(delayed_error_threshold) or delayed_error_threshold < 0:
            raise ValueError("delayed_error_threshold must be non-negative")
        if delayed_error_consecutive < 1 or delayed_error_teacher_rounds < 1:
            raise ValueError("delayed-error recovery values must be positive")
        if not np.isfinite(video_motion_gate_threshold) or video_motion_gate_threshold < 0:
            raise ValueError("video_motion_gate_threshold must be non-negative")
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
        self.delayed_error_threshold = float(delayed_error_threshold)
        self.delayed_error_consecutive = int(delayed_error_consecutive)
        self.delayed_error_teacher_rounds = int(delayed_error_teacher_rounds)
        self.video_motion_gate_threshold = float(video_motion_gate_threshold)
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
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=True) + "\n")

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
            self.pending_teacher_cache_updates.popleft()
            replayed += 1
        return replayed

    def _reset(self, request: dict) -> dict:
        start = time.perf_counter()
        self._reset_state()
        self._call(self.draft, request)
        response = self._call(self.teacher, request)
        self._log(source="reset", elapsed_sec=time.perf_counter() - start)
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
        if source == "full":
            self._call(self.teacher, dict(request))
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
            accepted_prefix=horizon,
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
        video_motion_stats = draft_response.get("video_motion_stats")
        horizon = action.shape[1] * action.shape[2]

        if self.video_motion_gate_threshold > 0:
            if video_motion_stats is None or "global_mean" not in video_motion_stats:
                raise RuntimeError("video motion gate requires draft global_mean")
            global_motion = float(video_motion_stats["global_mean"])
            if not np.isfinite(global_motion):
                raise ValueError("draft global video motion must be finite")
            if global_motion >= self.video_motion_gate_threshold:
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
                    video_motion_stats=video_motion_stats,
                    video_motion_gate_threshold=self.video_motion_gate_threshold,
                    elapsed_sec=time.perf_counter() - start,
                )
                return response

        verify_request = dict(request)
        verify_request.update(
            verify_action=True,
            action_latent=action_latent,
            verify_noise=self.rng.standard_normal(action_latent.shape).astype(np.float32),
            threshold=self.threshold,
            tau_timesteps=self.tau_timesteps,
            frame_st_id=self.frame_st_id,
            gripper_consensus=self.gripper_consensus,
        )
        if self.last_gripper is not None:
            previous_phase = self.last_gripper >= self.gripper_threshold
            verify_request["previous_gripper"] = np.where(
                previous_phase, 1.0, -1.0
            ).astype(np.float32)
        verify_response = self._call(self.teacher, verify_request)
        verified_prefix = self._accepted_prefix(action_latent, verify_response, horizon)
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
            "gripper_consensus_prefix": verify_response.get(
                "gripper_consensus_prefix"
            ),
            "gripper_consensus_failure_index": verify_response.get(
                "gripper_consensus_failure_index"
            ),
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

        switch_step = first_gripper_switch(
            action,
            self.gripper_channels,
            self.gripper_threshold,
            previous=self.last_gripper,
        )
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

        self.round_id += 1
        if accepted_prefix == 0:
            reason = self.force_full_reason or "zero_prefix"
            self.force_full_reason = reason
            self.last_source = "replan"
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
                teacher_gripper_switch=teacher_gripper_switch,
                **verify_telemetry,
                elapsed_sec=time.perf_counter() - start,
            )
            return response

        self.pending_cache_source = "flash"
        self.last_source = "draft_flash"
        self.flash_rounds_since_full += 1
        flow_budget_charge = self._flow_budget_charge(
            verify_response.get("distances"), accepted_prefix
        )
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
        self._log(
            source=self.last_source,
            fallback_reason=response.get("fallback_reason"),
            accepted_prefix=accepted_prefix,
            verified_prefix=verified_prefix,
            teacher_gripper_switch=teacher_gripper_switch,
            decoded_gripper_switch_step=switch_step,
            flow_budget_charge=flow_budget_charge,
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
