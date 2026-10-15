"""Training-free resolution-sensitivity probe for LingBot-VA on RoboTwin.

Question this probe answers before any foveated-prediction training: does the
policy actually depend on fine spatial detail during precision-demanding
("fine") chunks, and is that detail carried by the global (head) view or the
local (wrist) views?

The probe wraps a client policy and degrades selected camera views in the
observations the server consumes (the reset frame and every compute_kv_cache
key-frame batch). Degradation is a down-then-up resize, so tensor shapes and
the server path are unchanged. Whether a batch is degraded depends on the
phase of the action chunk that was just executed: key frames of chunk t
condition the prediction of chunk t+1, so the executed chunk is used as the
phase proxy for the next decision.

Phase rules:
  gripper: a gripper channel switches open/closed inside the chunk.
  speed:   mean end-effector translation per step is below a threshold
           (slow segments are precision segments, cf. AttenA+).
  either:  gripper or speed.
"""

import copy
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

try:
    import cv2
except ImportError:  # unit tests and analysis do not need OpenCV
    cv2 = None

from evaluation.robotwin.specverify_client_policy import has_gripper_switch

VIEW_KEYS = {
    "head": ("observation.images.cam_high",),
    "wrist": ("observation.images.cam_left_wrist", "observation.images.cam_right_wrist"),
    "all": (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ),
}
PHASES = ("all", "fine", "coarse")
RULES = ("gripper", "speed", "either")

# Translation channels of the 16-dim end-effector action used by RoboTwin:
# [x, y, z, qx, qy, qz, qw, grip] for the left arm, then the same for the right arm.
_TRANS_CHANNELS = ((0, 1, 2), (8, 9, 10))


def degrade_image(image: np.ndarray, factor: int) -> np.ndarray:
    """Remove detail finer than `factor` pixels while keeping shape and dtype."""
    if factor <= 1:
        return image
    h, w = image.shape[:2]
    small_h, small_w = max(1, h // factor), max(1, w // factor)
    if cv2 is not None:
        small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return out.astype(image.dtype, copy=False)
    # numpy fallback: block mean, then nearest upsampling
    hh, ww = small_h * factor, small_w * factor
    crop = image[:hh, :ww].astype(np.float32)
    blocks = crop.reshape(small_h, factor, small_w, factor, -1).mean(axis=(1, 3))
    up = np.repeat(np.repeat(blocks, factor, axis=0), factor, axis=1)
    out = image.astype(np.float32).copy()
    out[:hh, :ww] = up.reshape(out[:hh, :ww].shape)
    return out.astype(image.dtype)


def chunk_translation_speed(action: np.ndarray) -> float:
    """Mean per-step end-effector translation of an executed (C, F, H) chunk."""
    if action is None or action.ndim != 3 or action.shape[0] < 16:
        return float("nan")
    flat = action.reshape(action.shape[0], -1)
    speeds = []
    for chans in _TRANS_CHANNELS:
        xyz = flat[list(chans)].T
        if xyz.shape[0] > 1:
            speeds.append(np.linalg.norm(np.diff(xyz, axis=0), axis=1).mean())
    return float(np.mean(speeds)) if speeds else float("nan")


def classify_chunk(action: Optional[np.ndarray], rule: str, speed_thresh: float) -> Dict:
    switch = bool(action is not None and has_gripper_switch(action))
    speed = chunk_translation_speed(action) if action is not None else float("nan")
    slow = bool(np.isfinite(speed) and speed < speed_thresh)
    fine = {"gripper": switch, "speed": slow, "either": switch or slow}[rule]
    return {"fine": fine, "gripper_switch": switch, "speed": speed}


class FoveaProbeClientPolicy:
    """Wraps a client policy and degrades views in the observations it sends."""

    def __init__(
        self,
        policy,
        view: str = "wrist",
        phase: str = "fine",
        factor: int = 4,
        rule: str = "either",
        speed_thresh: float = 0.003,
        log_path: Optional[str] = None,
        condition: Optional[str] = None,
    ):
        if view not in VIEW_KEYS:
            raise ValueError(f"view must be one of {sorted(VIEW_KEYS)}")
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        if rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}")
        self.policy = policy
        self.view = view
        self.phase = phase
        self.factor = int(factor)
        self.rule = rule
        self.speed_thresh = float(speed_thresh)
        self.condition = condition or f"{view}-{phase}-x{self.factor}-{rule}"
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.episode = -1
        self._episode_stats = None

    # -- policy interface -------------------------------------------------
    def infer(self, obs: Dict) -> Dict:
        if obs.get("reset", False):
            self.episode += 1
            self._episode_stats = {"chunks": 0, "fine_chunks": 0, "degraded_batches": 0}
            return self.policy.infer(obs)
        if obs.get("compute_kv_cache", False):
            info = classify_chunk(obs.get("state"), self.rule, self.speed_thresh)
            degrade = self._should_degrade(info["fine"])
            self._record(info, degrade)
            if degrade:
                obs = dict(obs)
                obs["obs"] = self._degrade_frames(obs["obs"])
            return self.policy.infer(obs)
        if "obs" in obs and self.phase == "all":
            # Prediction requests carry the initial frame; it is only consumed at
            # frame 0, before any chunk has a phase, so only `all` degrades it.
            obs = dict(obs)
            obs["obs"] = self._degrade_frames(obs["obs"])
        return self.policy.infer(obs)

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def end_episode(self, success: bool, seed: Optional[int] = None) -> None:
        self._write({"event": "episode", "success": bool(success), "seed": seed,
                     **(self._episode_stats or {})})

    def __getattr__(self, name):
        return getattr(self.policy, name)

    # -- internals ----------------------------------------------------------
    def _should_degrade(self, fine: bool) -> bool:
        if self.phase == "all":
            return True
        return fine if self.phase == "fine" else not fine

    def _degrade_frames(self, frames):
        single = isinstance(frames, dict)
        out = []
        for frame in ([frames] if single else frames):
            frame = copy.copy(frame)  # shallow copy: never mutate the client's video buffers
            for key in VIEW_KEYS[self.view]:
                if key in frame:
                    frame[key] = degrade_image(np.asarray(frame[key]), self.factor)
            out.append(frame)
        return out[0] if single else out

    def _record(self, info: Dict, degraded: bool) -> None:
        if self._episode_stats is not None:
            self._episode_stats["chunks"] += 1
            self._episode_stats["fine_chunks"] += int(info["fine"])
            self._episode_stats["degraded_batches"] += int(degraded)
        self._write({"event": "chunk", "degraded": degraded, **info})

    def _write(self, record: Dict) -> None:
        if not self.log_path:
            return
        record = {"time": time.time(), "condition": self.condition, "episode": self.episode,
                  "view": self.view, "phase": self.phase, "factor": self.factor,
                  "rule": self.rule, **record}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=float) + "\n")


def load_records(paths: Iterable[str]):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
