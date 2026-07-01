#!/usr/bin/env python3
"""Offline teacher-self sanity for action-flow verification.

This is an engineering-alignment check, not a RobotWin success-rate run.
It asks whether a teacher-generated normalized action chunk is accepted by the
same teacher action flow at verifier timesteps such as 150 and 300.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "wan_va"))

from configs import VA_CONFIGS  # noqa: E402
from utils import init_logger  # noqa: E402
from wan_va_server import VA_Server  # noqa: E402


def load_example_robotwin_obs(config) -> dict:
    image_dir = REPO / "example" / "robotwin"
    image_dict = {}
    for key in config.obs_cam_keys:
        image_dict[key] = np.array(
            Image.open(image_dir / f"{key}.png").convert("RGB")
        )

    state = np.zeros(
        (len(config.used_action_channel_ids),
         config.frame_chunk_size,
         config.action_per_frame),
        dtype=np.float32,
    )
    return {"obs": [image_dict], "state": state}


def summarize_verify(result: dict, threshold: float) -> dict:
    distances = result["distances"]
    conditioned = int(result["conditioned_frame_count"])
    valid = distances[:, conditioned:, :]
    pass_mask = valid <= threshold
    return {
        "accepted_prefix": int(result["accepted_prefix"]),
        "raw_valid_prefix": int(result["raw_valid_prefix"]),
        "conditioned_frame_count": conditioned,
        "tau_timesteps": [float(x) for x in result["tau_timesteps"].tolist()],
        "threshold": float(result["threshold"]),
        "valid_step_count": int(valid.numel()),
        "pass_count": int(pass_mask.sum().item()),
        "pass_rate": float(pass_mask.float().mean().item()) if valid.numel() else 0.0,
        "all_valid_pass": bool(pass_mask.all().item()) if valid.numel() else False,
        "distance_mean": float(valid.mean().item()) if valid.numel() else 0.0,
        "distance_max": float(valid.max().item()) if valid.numel() else 0.0,
        "distance_p95": float(torch.quantile(valid.flatten(), 0.95).item())
        if valid.numel()
        else 0.0,
        "distance_matrix": distances.float().tolist(),
    }


def write_summary(run_root: Path, summary: dict) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "result.json").write_text(json.dumps(summary, indent=2))
    verdict = "PASS" if summary["verify"]["all_valid_pass"] else "FAIL"
    lines = [
        "# SpecVerify Teacher-Self Sanity",
        "",
        f"- Verdict: {verdict}",
        f"- Model: `{summary['model_path']}`",
        f"- GPU: `{summary['gpu']}`",
        f"- Tau timesteps: `{summary['verify']['tau_timesteps']}`",
        f"- Threshold: `{summary['verify']['threshold']}`",
        f"- Accepted prefix: `{summary['verify']['accepted_prefix']}`",
        f"- Pass rate: `{summary['verify']['pass_rate']:.4f}`",
        f"- Distance mean/max/p95: "
        f"`{summary['verify']['distance_mean']:.6f}` / "
        f"`{summary['verify']['distance_max']:.6f}` / "
        f"`{summary['verify']['distance_p95']:.6f}`",
        "",
        "This is an engineering sanity check only. It does not measure RobotWin success.",
    ]
    (run_root / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--tau", type=float, nargs="+", default=[150.0, 300.0])
    parser.add_argument("--prompt", default=VA_CONFIGS["robotwin_i2av"].prompt)
    parser.add_argument("--enable-offload", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    init_logger()

    config = copy.deepcopy(VA_CONFIGS["robotwin"])
    config.wan22_pretrained_model_name_or_path = args.model_path
    config.save_root = str(Path(args.run_root) / "server")
    config.local_rank = 0
    config.rank = 0
    config.world_size = 1
    config.enable_offload = bool(args.enable_offload)

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    start = time.time()
    model = VA_Server(config)
    model._reset(prompt=args.prompt)
    obs = load_example_robotwin_obs(config)
    model._infer(
        obs,
        frame_st_id=0,
        teacher_self_verify={
            "tau_timesteps": tuple(args.tau),
            "threshold": args.threshold,
        },
    )
    result = model.last_teacher_self_verify
    if result is None:
        raise RuntimeError("teacher_self_verify did not produce a result")

    summary = {
        "status": "completed",
        "model_path": args.model_path,
        "run_root": str(run_root),
        "gpu": str(args.gpu),
        "prompt": args.prompt,
        "elapsed_sec": time.time() - start,
        "verify": summarize_verify(result, args.threshold),
    }
    write_summary(run_root, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
