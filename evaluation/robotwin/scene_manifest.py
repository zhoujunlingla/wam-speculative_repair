"""Record and validate deterministic RoboTwin evaluation scenes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1


def _pose_values(pose) -> dict:
    return {
        "p": np.asarray(pose.p, dtype=np.float64).round(6).tolist(),
        "q": np.asarray(pose.q, dtype=np.float64).round(6).tolist(),
    }


def scene_fingerprint(task_env) -> str:
    """Hash actor/articulation state after ``setup_demo`` and settling."""

    actors = []
    for index, actor in enumerate(task_env.scene.get_all_actors()):
        actors.append({
            "index": index,
            "name": actor.get_name(),
            "pose": _pose_values(actor.get_pose()),
        })
    articulations = []
    for index, articulation in enumerate(task_env.scene.get_all_articulations()):
        articulations.append({
            "index": index,
            "name": articulation.get_name(),
            "pose": _pose_values(articulation.get_root_pose()),
            "qpos": np.asarray(
                articulation.get_qpos(), dtype=np.float64
            ).round(6).tolist(),
        })
    payload = json.dumps(
        {"actors": actors, "articulations": articulations},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def robotwin_provenance(root: Path) -> dict:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    tracked_diff = git("diff", "--binary", "HEAD")
    return {
        "root": str(root.resolve()),
        "commit": git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "tracked_diff_sha256": (
            None if tracked_diff is None else
            hashlib.sha256(tracked_diff.encode()).hexdigest()
        ),
    }


def load_manifest(path: Path, *, task_name: str, task_config: str,
                  test_num: int) -> dict:
    manifest = json.loads(path.read_text())
    expected = {
        "schema_version": SCHEMA_VERSION,
        "task_name": task_name,
        "task_config": task_config,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"scene manifest {key}={manifest.get(key)!r}, expected {value!r}"
            )
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) < test_num:
        raise ValueError("scene manifest does not contain enough episodes")
    return manifest


def write_manifest_atomic(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
