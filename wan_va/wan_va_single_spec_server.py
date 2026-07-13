# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Serve draft and teacher VA models behind one realtime-flash policy."""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.robotwin.realtime_flash_policy import RealtimeFlashPolicy
from wan_va.configs import VA_CONFIGS
from wan_va.utils.Simple_Remote_Infer.deploy.websocket_policy_server import (
    WebsocketPolicyServer,
)
from wan_va.wan_va_server import VA_Server


class LocalModelAdapter:
    """Expose a local ``VA_Server`` through the policy's small infer contract."""

    def __init__(self, model: VA_Server) -> None:
        self.model = model

    def infer(self, request: dict) -> dict:
        verifying = bool(request.get("verify_action", False))
        frame_st_id = getattr(self.model, "frame_st_id", None) if verifying else None
        response = self.model.infer(request)
        if verifying and getattr(self.model, "frame_st_id", None) != frame_st_id:
            raise RuntimeError("teacher verification mutated frame_st_id")
        return response


def _model_config(name: str, save_root: Path, local_rank: int):
    config = copy.deepcopy(VA_CONFIGS[name])
    config.save_root = str(save_root)
    config.rank = 0
    config.local_rank = local_rank
    config.world_size = 1
    return config


def build_policy(args: argparse.Namespace):
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "draft_only":
        logging.info("loading local draft model: %s", args.draft_config_name)
        return LocalModelAdapter(VA_Server(
            _model_config(args.draft_config_name, save_root / "draft", args.local_rank)
        ))
    if args.mode == "teacher_only":
        logging.info("loading local teacher model: %s", args.teacher_config_name)
        return LocalModelAdapter(VA_Server(
            _model_config(args.teacher_config_name, save_root / "teacher", args.local_rank)
        ))
    logging.info("loading local draft model: %s", args.draft_config_name)
    draft = VA_Server(_model_config(
        args.draft_config_name, save_root / "draft", args.local_rank
    ))
    logging.info("loading local teacher model: %s", args.teacher_config_name)
    teacher = VA_Server(
        _model_config(args.teacher_config_name, save_root / "teacher", args.local_rank)
    )
    return RealtimeFlashPolicy(
        LocalModelAdapter(draft),
        LocalModelAdapter(teacher),
        pf_interval=args.pf_interval,
        threshold=args.threshold,
        tau_timesteps=args.tau_timesteps,
        teacher_gripper_fallback=args.teacher_gripper_fallback,
        flow_budget_threshold=args.flow_budget_threshold,
        flow_budget_burst_after=args.flow_budget_burst_after,
        flow_budget_burst_rounds=args.flow_budget_burst_rounds,
        flow_budget_burst_limit=args.flow_budget_burst_limit,
        flow_budget_motion_ceiling=args.flow_budget_motion_ceiling,
        delayed_error_threshold=args.delayed_error_threshold,
        delayed_error_consecutive=args.delayed_error_consecutive,
        delayed_error_teacher_rounds=args.delayed_error_teacher_rounds,
        video_motion_gate_threshold=args.video_motion_gate_threshold,
        gripper_full_window=args.gripper_full_window,
        gripper_consensus=args.gripper_consensus,
        rng=None if args.seed is None else np.random.default_rng(args.seed),
        log_path=args.log_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-config-name", default="robotwin_flashwam_step3000_v1a2")
    parser.add_argument("--teacher-config-name", default="robotwin_lingbot_v2a4")
    parser.add_argument("--save-root", default="/tmp/wanva_realtime_flash")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=("spec", "draft_only", "teacher_only"),
        default="spec",
    )
    parser.add_argument(
        "--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0))
    )
    parser.add_argument("--pf-interval", type=int, default=2)
    parser.add_argument(
        "--teacher-gripper-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--flow-budget-threshold", type=float, default=0.0)
    parser.add_argument("--flow-budget-burst-after", type=int, default=0)
    parser.add_argument("--flow-budget-burst-rounds", type=int, default=0)
    parser.add_argument("--flow-budget-burst-limit", type=int, default=0)
    parser.add_argument("--flow-budget-motion-ceiling", type=float, default=0.0)
    parser.add_argument("--delayed-error-threshold", type=float, default=0.0)
    parser.add_argument("--delayed-error-consecutive", type=int, default=2)
    parser.add_argument("--delayed-error-teacher-rounds", type=int, default=2)
    parser.add_argument("--video-motion-gate-threshold", type=float, default=0.0)
    parser.add_argument("--gripper-full-window", type=int, default=1)
    parser.add_argument("--gripper-consensus", action="store_true")
    parser.add_argument(
        "--tau-timesteps", type=float, nargs="+", default=(50.0, 100.0)
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-path", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    args = parse_args()
    policy = build_policy(args)
    server = WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata={
            "server": "wanva_realtime_flash",
            "mode": args.mode,
            "draft_config": args.draft_config_name,
            "teacher_config": args.teacher_config_name,
        },
    )
    logging.info("serving realtime-flash policy on %s:%s", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
