import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def slice_action_prefix(action: np.ndarray, prefix_len: int) -> np.ndarray:
    """Slice a [C, F, H] action chunk at model frame boundaries."""

    if action.ndim != 3:
        raise ValueError("action must have shape [C, F, H]")
    action_per_frame = action.shape[2]
    frame_count = max(0, min(prefix_len // action_per_frame, action.shape[1]))
    return action[:, :frame_count, :]


def has_gripper_switch(action: np.ndarray, threshold: float = 0.5) -> bool:
    """Detect Realtime-VLA-style phase transition from gripper channels."""

    if action.ndim != 3 or action.shape[0] < 16:
        return False
    for channel in (7, 15):
        values = action[channel].reshape(-1)
        if values.size > 1 and (values[0] > threshold) != (values[-1] > threshold):
            return True
    return False


class SpecVerifyClientPolicy:
    """Two-server speculative policy: FlashWAM draft + LingBot teacher verifier."""

    def __init__(
        self,
        draft_port: int,
        teacher_port: int,
        host: str = "0.0.0.0",
        pf_interval: int = 2,
        threshold: float = 0.15,
        tau_timesteps=(150.0, 300.0),
        teacher_cache_mode: str = "sync",
        prime_draft_on_teacher_full: bool = True,
        phase_mode: str = "fallback",
        phase_threshold_scale: float = 0.5,
        log_path: Optional[str] = None,
        client_factory=None,
    ) -> None:
        if client_factory is None:
            from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
            client_factory = WebsocketClientPolicy
        self.draft = client_factory(host=host, port=draft_port)
        self.teacher = client_factory(host=host, port=teacher_port)
        self.pf_interval = int(pf_interval)
        self.threshold = float(threshold)
        self.tau_timesteps = tuple(float(x) for x in tau_timesteps)
        if teacher_cache_mode not in ("sync", "lazy_reference"):
            raise ValueError("teacher_cache_mode must be 'sync' or 'lazy_reference'")
        if phase_mode not in ("fallback", "tighten", "ignore"):
            raise ValueError("phase_mode must be 'fallback', 'tighten', or 'ignore'")
        self.teacher_cache_mode = teacher_cache_mode
        self.prime_draft_on_teacher_full = bool(prime_draft_on_teacher_full)
        self.phase_mode = phase_mode
        self.phase_threshold_scale = float(phase_threshold_scale)
        self.round_id = 0
        self.frame_st_id = 0
        self.flash_rounds_since_full = 0
        self.last_source = None
        self.pending_teacher_cache_obs = []
        self.pending_teacher_cache_frames = 0
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.latency_log_path = Path(os.environ["WANVA_POLICY_LATENCY_LOG"]) if os.environ.get("WANVA_POLICY_LATENCY_LOG") else None
        if self.latency_log_path:
            self.latency_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, record: Dict) -> None:
        record = dict(record)
        record.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
        record.setdefault("round_id", self.round_id)
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _log_latency(self, obs: Dict, elapsed: float, source: str) -> None:
        if not self.latency_log_path:
            return
        if obs.get("reset", False):
            request_type = "reset"
        elif obs.get("compute_kv_cache", False):
            request_type = "compute_kv_cache"
        else:
            request_type = "action"
        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "policy": "specverify",
            "request_type": request_type,
            "source": source,
            "round_id": self.round_id,
            "elapsed_sec": elapsed,
        }
        with self.latency_log_path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _need_full_path(self) -> bool:
        return self.round_id == 0 or self.flash_rounds_since_full >= self.pf_interval

    @staticmethod
    def _state_frame_count(obs: Dict) -> int:
        state = obs.get("state")
        if isinstance(state, np.ndarray) and state.ndim >= 2:
            return int(state.shape[1])
        return 0

    @staticmethod
    def _copy_cache_obs(obs: Dict) -> Dict:
        # The RobotWin client creates a fresh obs dict for each cache update. A
        # shallow copy is enough to preserve the references until the next PF or
        # fallback sync without duplicating camera tensors.
        return dict(obs)

    def _sync_teacher_pending(self) -> None:
        if self.teacher_cache_mode != "lazy_reference":
            return
        if not self.pending_teacher_cache_obs:
            return
        start = time.perf_counter()
        pending_count = len(self.pending_teacher_cache_obs)
        pending_frames = self.pending_teacher_cache_frames
        for pending_obs in self.pending_teacher_cache_obs:
            self.teacher.infer(pending_obs)
        self.pending_teacher_cache_obs = []
        self.pending_teacher_cache_frames = 0
        self._log({
            "source": "teacher_cache_sync",
            "synced_cache_updates": pending_count,
            "synced_frames": pending_frames,
            "elapsed_sec": time.perf_counter() - start,
        })

    def infer(self, obs: Dict) -> Dict:
        start = time.perf_counter()
        if obs.get("reset", False):
            self.round_id = 0
            self.frame_st_id = 0
            self.flash_rounds_since_full = 0
            self.last_source = None
            self.pending_teacher_cache_obs = []
            self.pending_teacher_cache_frames = 0
            self.draft.infer(obs)
            ret = self.teacher.infer(obs)
            self._log_latency(obs, time.perf_counter() - start, "reset")
            return ret

        if obs.get("compute_kv_cache", False):
            frame_count = self._state_frame_count(obs)
            # Realtime-VLA-style lazy reference mode keeps the teacher verifier
            # on the last full-path reference cache during draft rounds. Real
            # history is synced only before a teacher/full path is actually
            # taken. The draft still receives every real cache update.
            if self.teacher_cache_mode == "sync":
                self.teacher.infer(obs)
            else:
                self.pending_teacher_cache_obs.append(self._copy_cache_obs(obs))
                self.pending_teacher_cache_frames += frame_count
            self.draft.infer(obs)
            self.frame_st_id += frame_count
            self._log({
                "source": "compute_kv_cache",
                "teacher_cache_mode": self.teacher_cache_mode,
                "pending_teacher_cache_updates": len(self.pending_teacher_cache_obs),
                "pending_teacher_cache_frames": self.pending_teacher_cache_frames,
                "frame_st_id": self.frame_st_id,
                "elapsed_sec": time.perf_counter() - start,
            })
            latency_source = "compute_kv_cache" if self.teacher_cache_mode == "sync" else "compute_kv_cache_lazy_reference"
            self._log_latency(obs, time.perf_counter() - start, latency_source)
            return {}

        if self._need_full_path():
            self._sync_teacher_pending()
            ret = self.teacher.infer(obs)
            # Full-path refresh executes the teacher action, but LingBot/FlashWAM
            # servers also keep streaming VAE and transformer caches. Prime the
            # draft cache with the same observation so the following real-history
            # cache update has the same temporal state as the teacher.
            if self.prime_draft_on_teacher_full:
                self.draft.infer(obs)
            self.last_source = "teacher_full"
            self.flash_rounds_since_full = 0
            self.round_id += 1
            elapsed = time.perf_counter() - start
            self._log({"source": self.last_source, "accepted_prefix": 32, "elapsed_sec": elapsed})
            self._log_latency(obs, elapsed, self.last_source)
            return ret

        draft_obs = dict(obs)
        draft_obs["return_action_latent"] = True
        draft_ret = self.draft.infer(draft_obs)
        draft_action = draft_ret["action"]

        phase_switch = has_gripper_switch(draft_action)
        if phase_switch and self.phase_mode == "fallback":
            self._sync_teacher_pending()
            ret = self.teacher.infer(obs)
            self.last_source = "teacher_fallback"
            self.flash_rounds_since_full = 0
            self.round_id += 1
            elapsed = time.perf_counter() - start
            self._log({
                "source": self.last_source,
                "fallback_reason": "phase_gripper_switch",
                "accepted_prefix": 0,
                "phase_mode": self.phase_mode,
                "elapsed_sec": elapsed,
            })
            self._log_latency(obs, elapsed, self.last_source)
            return ret

        verify_threshold = self.threshold
        if phase_switch and self.phase_mode == "tighten":
            verify_threshold = self.threshold * self.phase_threshold_scale

        verify_ret = self.teacher.infer({
            "verify_action": True,
            "action_latent": draft_ret["action_latent"],
            "threshold": verify_threshold,
            "tau_timesteps": self.tau_timesteps,
            "frame_st_id": self.frame_st_id,
        })
        verify_ret["teacher_cache_mode"] = self.teacher_cache_mode
        verify_ret["pending_teacher_cache_updates"] = len(self.pending_teacher_cache_obs)
        verify_ret["pending_teacher_cache_frames"] = self.pending_teacher_cache_frames
        verify_ret["phase_switch"] = phase_switch
        verify_ret["phase_mode"] = self.phase_mode
        verify_ret["base_threshold"] = self.threshold
        verify_ret["effective_threshold"] = verify_threshold
        accepted_prefix = int(verify_ret.get("accepted_prefix", 0))
        if accepted_prefix <= 0:
            self._sync_teacher_pending()
            ret = self.teacher.infer(obs)
            self.last_source = "teacher_fallback"
            self.flash_rounds_since_full = 0
            self.round_id += 1
            elapsed = time.perf_counter() - start
            self._log({
                "source": self.last_source,
                "fallback_reason": "verify_reject_phase_tightened" if phase_switch and self.phase_mode == "tighten" else "verify_reject",
                "accepted_prefix": accepted_prefix,
                "verify": verify_ret,
                "elapsed_sec": elapsed,
            })
            self._log_latency(obs, elapsed, self.last_source)
            return ret

        action = slice_action_prefix(draft_action, accepted_prefix)
        self.last_source = "draft"
        self.flash_rounds_since_full += 1
        self.round_id += 1
        elapsed = time.perf_counter() - start
        self._log({
            "source": self.last_source,
            "accepted_prefix": accepted_prefix,
            "verify": verify_ret,
            "elapsed_sec": elapsed,
        })
        self._log_latency(obs, elapsed, self.last_source)
        return {"action": action}


def _action_sequence(action: np.ndarray) -> np.ndarray:
    if not isinstance(action, np.ndarray) or action.ndim != 3:
        return np.zeros((0, 0), dtype=np.float32)
    # LingBot action chunks are [C, F, H]. Convert to chronological [F*H, C].
    return action.transpose(1, 2, 0).reshape(-1, action.shape[0])


def action_risk_score(
    action: np.ndarray,
    *,
    delta_ref: float = 0.25,
    mean_delta_ref: float = 0.12,
    jerk_ref: float = 0.18,
    phase_weight: float = 0.25,
) -> Dict[str, object]:
    seq = _action_sequence(action)
    metrics: Dict[str, object] = {
        "risk_score": 0.0,
        "max_step_delta": 0.0,
        "mean_step_delta": 0.0,
        "max_jerk": 0.0,
        "phase_switch": False,
    }
    if seq.size == 0:
        metrics["risk_score"] = 1.0
        metrics["reason"] = "bad_action_shape"
        return metrics

    cont_idx = [i for i in range(seq.shape[1]) if i not in (7, 15)]
    cont = seq[:, cont_idx] if cont_idx else seq
    if len(cont) >= 2:
        delta = np.linalg.norm(np.diff(cont, axis=0), axis=1)
        metrics["max_step_delta"] = float(delta.max(initial=0.0))
        metrics["mean_step_delta"] = float(delta.mean() if delta.size else 0.0)
    if len(cont) >= 3:
        jerk = np.linalg.norm(np.diff(cont, n=2, axis=0), axis=1)
        metrics["max_jerk"] = float(jerk.max(initial=0.0))

    phase_switch = has_gripper_switch(action)
    metrics["phase_switch"] = bool(phase_switch)
    score = 0.0
    score += 0.30 * min(float(metrics["max_step_delta"]) / max(delta_ref, 1e-6), 1.0)
    score += 0.20 * min(float(metrics["mean_step_delta"]) / max(mean_delta_ref, 1e-6), 1.0)
    score += 0.25 * min(float(metrics["max_jerk"]) / max(jerk_ref, 1e-6), 1.0)
    if phase_switch:
        score += phase_weight
    metrics["risk_score"] = float(min(score, 1.0))
    return metrics


class RiskRouterClientPolicy:
    """Route LingBot v1/a2 draft to LingBot v2/a4 only when the chunk looks risky."""

    def __init__(
        self,
        draft_port: int,
        teacher_port: int,
        host: str = "0.0.0.0",
        threshold: float = 0.20,
        tau_timesteps=(150.0, 300.0),
        teacher_cache_mode: str = "lazy_reference",
        risk_low: float = 0.45,
        risk_high: float = 0.80,
        risk_verify_mode: str = "medium",
        risk_delta_ref: float = 0.25,
        risk_mean_delta_ref: float = 0.12,
        risk_jerk_ref: float = 0.18,
        risk_phase_weight: float = 0.25,
        phase_threshold_scale: float = 0.5,
        world_verify_enable: bool = False,
        world_verify_threshold: float = 0.35,
        world_verify_tau_timesteps=(150.0, 300.0),
        log_path: Optional[str] = None,
        client_factory=None,
    ) -> None:
        if client_factory is None:
            from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
            client_factory = WebsocketClientPolicy
        if teacher_cache_mode not in ("sync", "lazy_reference", "stale_reference"):
            raise ValueError("teacher_cache_mode must be 'sync', 'lazy_reference', or 'stale_reference'")
        if risk_verify_mode not in ("off", "medium"):
            raise ValueError("risk_verify_mode must be 'off' or 'medium'")
        self.draft = client_factory(host=host, port=draft_port)
        self.teacher = client_factory(host=host, port=teacher_port)
        self.threshold = float(threshold)
        self.tau_timesteps = tuple(float(x) for x in tau_timesteps)
        self.teacher_cache_mode = teacher_cache_mode
        self.risk_low = float(risk_low)
        self.risk_high = float(risk_high)
        self.risk_verify_mode = risk_verify_mode
        self.risk_delta_ref = float(risk_delta_ref)
        self.risk_mean_delta_ref = float(risk_mean_delta_ref)
        self.risk_jerk_ref = float(risk_jerk_ref)
        self.risk_phase_weight = float(risk_phase_weight)
        self.phase_threshold_scale = float(phase_threshold_scale)
        self.world_verify_enable = bool(world_verify_enable)
        self.world_verify_threshold = float(world_verify_threshold)
        self.world_verify_tau_timesteps = tuple(float(x) for x in world_verify_tau_timesteps)
        self.round_id = 0
        self.frame_st_id = 0
        self.pending_teacher_cache_obs = []
        self.pending_teacher_cache_frames = 0
        self.latest_teacher_cache_obs = None
        self.latest_teacher_cache_frames = 0
        self.latest_teacher_cache_synced = True
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.latency_log_path = Path(os.environ["WANVA_POLICY_LATENCY_LOG"]) if os.environ.get("WANVA_POLICY_LATENCY_LOG") else None
        if self.latency_log_path:
            self.latency_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, record: Dict) -> None:
        record = dict(record)
        record.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
        record.setdefault("round_id", self.round_id)
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _log_latency(self, obs: Dict, elapsed: float, source: str) -> None:
        if not self.latency_log_path:
            return
        if obs.get("reset", False):
            request_type = "reset"
        elif obs.get("compute_kv_cache", False):
            request_type = "compute_kv_cache"
        else:
            request_type = "action"
        with self.latency_log_path.open("a") as f:
            f.write(json.dumps({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "policy": "risk_router",
                "request_type": request_type,
                "source": source,
                "round_id": self.round_id,
                "elapsed_sec": elapsed,
            }, ensure_ascii=False) + "\n")

    @staticmethod
    def _state_frame_count(obs: Dict) -> int:
        state = obs.get("state")
        if isinstance(state, np.ndarray) and state.ndim >= 2:
            return int(state.shape[1])
        return 0

    @staticmethod
    def _copy_cache_obs(obs: Dict) -> Dict:
        return dict(obs)

    def _sync_teacher_pending(self) -> None:
        if self.teacher_cache_mode != "lazy_reference" or not self.pending_teacher_cache_obs:
            return
        start = time.perf_counter()
        pending_count = len(self.pending_teacher_cache_obs)
        pending_frames = self.pending_teacher_cache_frames
        for pending_obs in self.pending_teacher_cache_obs:
            self.teacher.infer(pending_obs)
        self.pending_teacher_cache_obs = []
        self.pending_teacher_cache_frames = 0
        self._log({
            "source": "teacher_cache_sync",
            "synced_cache_updates": pending_count,
            "synced_frames": pending_frames,
            "elapsed_sec": time.perf_counter() - start,
        })

    def _sync_teacher_latest(self) -> None:
        if self.teacher_cache_mode != "stale_reference":
            return
        if self.latest_teacher_cache_obs is None or self.latest_teacher_cache_synced:
            return
        start = time.perf_counter()
        self.teacher.infer(self.latest_teacher_cache_obs)
        self.latest_teacher_cache_synced = True
        self._log({
            "source": "teacher_cache_latest_sync",
            "synced_cache_updates": 1,
            "synced_frames": self.latest_teacher_cache_frames,
            "elapsed_sec": time.perf_counter() - start,
        })

    def _teacher_action(self, obs: Dict, source: str, extra: Dict) -> Dict:
        pre_teacher_elapsed = float(extra.pop("elapsed_sec", 0.0))
        teacher_start = time.perf_counter()
        self._sync_teacher_pending()
        self._sync_teacher_latest()
        ret = self.teacher.infer(obs)
        elapsed = pre_teacher_elapsed + (time.perf_counter() - teacher_start)
        record = {"source": source, "accepted_prefix": 32, "elapsed_sec": elapsed}
        record.update(extra)
        self._log(record)
        self._log_latency(obs, elapsed, source)
        return ret

    def infer(self, obs: Dict) -> Dict:
        start = time.perf_counter()
        if obs.get("reset", False):
            self.round_id = 0
            self.frame_st_id = 0
            self.pending_teacher_cache_obs = []
            self.pending_teacher_cache_frames = 0
            self.latest_teacher_cache_obs = None
            self.latest_teacher_cache_frames = 0
            self.latest_teacher_cache_synced = True
            self.draft.infer(obs)
            ret = self.teacher.infer(obs)
            self._log_latency(obs, time.perf_counter() - start, "reset")
            return ret

        if obs.get("compute_kv_cache", False):
            frame_count = self._state_frame_count(obs)
            if self.teacher_cache_mode == "sync":
                self.teacher.infer(obs)
                cache_source = "compute_kv_cache"
            elif self.teacher_cache_mode == "lazy_reference":
                self.pending_teacher_cache_obs.append(self._copy_cache_obs(obs))
                self.pending_teacher_cache_frames += frame_count
                cache_source = "compute_kv_cache_lazy_reference"
            else:
                self.latest_teacher_cache_obs = self._copy_cache_obs(obs)
                self.latest_teacher_cache_frames = frame_count
                self.latest_teacher_cache_synced = False
                cache_source = "compute_kv_cache_stale_reference"
            self.draft.infer(obs)
            self.frame_st_id += frame_count
            elapsed = time.perf_counter() - start
            self._log({
                "source": cache_source,
                "teacher_cache_mode": self.teacher_cache_mode,
                "pending_teacher_cache_updates": len(self.pending_teacher_cache_obs),
                "pending_teacher_cache_frames": self.pending_teacher_cache_frames,
                "latest_teacher_cache_frames": self.latest_teacher_cache_frames,
                "latest_teacher_cache_synced": self.latest_teacher_cache_synced,
                "frame_st_id": self.frame_st_id,
                "elapsed_sec": elapsed,
            })
            self._log_latency(obs, elapsed, cache_source)
            return {}

        draft_obs = dict(obs)
        draft_obs["return_action_latent"] = self.risk_verify_mode == "medium"
        draft_obs["return_video_latent"] = self.world_verify_enable
        draft_ret = self.draft.infer(draft_obs)
        draft_action = draft_ret["action"]
        risk = action_risk_score(
            draft_action,
            delta_ref=self.risk_delta_ref,
            mean_delta_ref=self.risk_mean_delta_ref,
            jerk_ref=self.risk_jerk_ref,
            phase_weight=self.risk_phase_weight,
        )
        risk_score = float(risk["risk_score"])
        elapsed = time.perf_counter() - start

        if self.frame_st_id == 0:
            self.round_id += 1
            return self._teacher_action(obs, "teacher_initial_prime", {
                "risk": risk,
                "accepted_prefix": 0,
                "elapsed_sec": elapsed,
                "pending_teacher_cache_updates": len(self.pending_teacher_cache_obs),
                "pending_teacher_cache_frames": self.pending_teacher_cache_frames,
            })

        high_risk = risk_score >= self.risk_high
        phase_switch = bool(risk.get("phase_switch"))
        if (risk_score < self.risk_low and not phase_switch) or self.risk_verify_mode == "off":
            source = "draft_low_risk" if risk_score < self.risk_low else "draft_medium_noverify"
            self._log({
                "source": source,
                "accepted_prefix": 32,
                "risk": risk,
                "pending_teacher_cache_updates": len(self.pending_teacher_cache_obs),
                "pending_teacher_cache_frames": self.pending_teacher_cache_frames,
                "elapsed_sec": elapsed,
            })
            self._log_latency(obs, elapsed, source)
            self.round_id += 1
            return {"action": draft_action}

        verify_threshold = self.threshold
        if phase_switch:
            verify_threshold = self.threshold * self.phase_threshold_scale

        verify_ret = self.teacher.infer({
            "verify_action": True,
            "action_latent": draft_ret["action_latent"],
            "threshold": verify_threshold,
            "tau_timesteps": self.tau_timesteps,
            "frame_st_id": self.frame_st_id,
        })
        verify_ret["phase_switch"] = phase_switch
        verify_ret["phase_mode"] = "tighten" if phase_switch else "normal"
        verify_ret["risk_zone"] = "high" if high_risk else "medium"
        verify_ret["base_threshold"] = self.threshold
        verify_ret["effective_threshold"] = verify_threshold
        accepted_prefix = int(verify_ret.get("accepted_prefix", 0))
        full_prefix = int(draft_action.shape[1] * draft_action.shape[2]) if getattr(draft_action, "ndim", 0) == 3 else 0
        action_per_frame = int(draft_action.shape[2]) if getattr(draft_action, "ndim", 0) == 3 else 0
        if self.frame_st_id == 0 and action_per_frame > 0 and 0 < accepted_prefix <= action_per_frame:
            verify_ret["raw_accepted_prefix"] = accepted_prefix
            verify_ret["initial_partial_prefix_rejected"] = True
            accepted_prefix = 0
        if self.teacher_cache_mode in ("stale_reference", "lazy_reference") and 0 < accepted_prefix < full_prefix:
            verify_ret["raw_accepted_prefix"] = accepted_prefix
            verify_ret[f"{self.teacher_cache_mode}_partial_prefix_rejected"] = True
            accepted_prefix = 0
        elapsed = time.perf_counter() - start
        if accepted_prefix <= 0:
            self.round_id += 1
            return self._teacher_action(obs, "teacher_verify_reject", {
                "risk": risk,
                "verify": verify_ret,
                "accepted_prefix": accepted_prefix,
                "elapsed_sec": elapsed,
                "pending_teacher_cache_updates": len(self.pending_teacher_cache_obs),
                "pending_teacher_cache_frames": self.pending_teacher_cache_frames,
            })

        world_verify_ret = None
        if self.world_verify_enable:
            video_latent = draft_ret.get("video_latent")
            if video_latent is None:
                raise RuntimeError("world verifier requires draft server return_video_latent support")
            world_verify_ret = self.teacher.infer({
                "verify_world_latent": True,
                "video_latent": video_latent,
                "threshold": self.world_verify_threshold,
                "tau_timesteps": self.world_verify_tau_timesteps,
                "frame_st_id": self.frame_st_id,
            })
            verify_ret["world_verify"] = world_verify_ret
            if not bool(world_verify_ret.get("world_pass", False)):
                elapsed = time.perf_counter() - start
                self.round_id += 1
                return self._teacher_action(obs, "teacher_world_verify_reject", {
                    "risk": risk,
                    "verify": verify_ret,
                    "accepted_prefix": 0,
                    "elapsed_sec": elapsed,
                    "pending_teacher_cache_updates": len(self.pending_teacher_cache_obs),
                    "pending_teacher_cache_frames": self.pending_teacher_cache_frames,
                })

        action = slice_action_prefix(draft_action, accepted_prefix)
        self._log({
            "source": "draft_verify_accept",
            "accepted_prefix": accepted_prefix,
            "risk": risk,
            "verify": verify_ret,
            "world_verify": world_verify_ret,
            "pending_teacher_cache_updates": len(self.pending_teacher_cache_obs),
            "pending_teacher_cache_frames": self.pending_teacher_cache_frames,
            "elapsed_sec": time.perf_counter() - start,
        })
        self._log_latency(obs, time.perf_counter() - start, "draft_verify_accept")
        self.round_id += 1
        return {"action": action}
