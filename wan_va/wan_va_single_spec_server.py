# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Single-process speculative LingBot server.

ponytail: this intentionally reuses RiskRouterClientPolicy by replacing the
websocket draft/teacher clients with local VA_Server adapters. It proves the
Realtime-VLA-style one-policy-server shape without forking the router logic.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs import VA_CONFIGS
from evaluation.robotwin.specverify_client_policy import RiskRouterClientPolicy
from utils.Simple_Remote_Infer.deploy.websocket_policy_server import WebsocketPolicyServer
from wan_va_server import VA_Server


class LocalModelClient:
    """Tiny adapter exposing a local VA_Server as a WebsocketClientPolicy lookalike."""

    def __init__(self, model: VA_Server) -> None:
        self.model = model

    def infer(self, obs):
        return self.model.infer(obs)


class LocalClientFactory:
    """Return draft then teacher local clients for RiskRouterClientPolicy."""

    def __init__(self, draft: VA_Server, teacher: VA_Server) -> None:
        self.clients = [LocalModelClient(draft), LocalModelClient(teacher)]
        self.index = 0

    def __call__(self, host="0.0.0.0", port=None):
        if self.index >= len(self.clients):
            raise RuntimeError("single spec server expected exactly draft and teacher clients")
        client = self.clients[self.index]
        self.index += 1
        return client


def _build_config(name: str, save_root: Path, local_rank: int):
    cfg = copy.deepcopy(VA_CONFIGS[name])
    cfg.save_root = str(save_root)
    cfg.local_rank = local_rank
    cfg.rank = 0
    cfg.world_size = 1
    return cfg


def build_policy(args: argparse.Namespace) -> RiskRouterClientPolicy:
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    draft_cfg = _build_config(args.draft_config_name, save_root / "draft", args.local_rank)
    teacher_cfg = _build_config(args.teacher_config_name, save_root / "teacher", args.local_rank)

    logging.info("loading draft config=%s", args.draft_config_name)
    draft = VA_Server(draft_cfg)
    logging.info("loading teacher config=%s", args.teacher_config_name)
    teacher = VA_Server(teacher_cfg)

    return RiskRouterClientPolicy(
        draft_port=0,
        teacher_port=0,
        host="local",
        threshold=args.threshold,
        tau_timesteps=args.tau_timesteps,
        teacher_cache_mode=args.teacher_cache_mode,
        risk_low=args.risk_low,
        risk_high=args.risk_high,
        risk_verify_mode=args.risk_verify_mode,
        risk_delta_ref=args.risk_delta_ref,
        risk_mean_delta_ref=args.risk_mean_delta_ref,
        risk_jerk_ref=args.risk_jerk_ref,
        risk_phase_weight=args.risk_phase_weight,
        phase_threshold_scale=args.phase_threshold_scale,
        world_verify_enable=args.world_verify_enable,
        world_verify_threshold=args.world_verify_threshold,
        world_verify_tau_timesteps=args.world_verify_tau_timesteps,
        repair_enable=args.repair_enable,
        repair_lambda=args.repair_lambda,
        repair_instrument_only=args.repair_instrument_only,
        svdr_repair_enable=args.svdr_repair_enable,
        svdr_lambda_min=args.svdr_lambda_min,
        svdr_lambda_max=args.svdr_lambda_max,
        svdr_motion_ref=args.svdr_motion_ref,
        svdr_topk_frac=args.svdr_topk_frac,
        svdr_temperature=args.svdr_temperature,
        log_path=args.specverify_log,
        client_factory=LocalClientFactory(draft, teacher),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve draft+teacher speculative policy in one process")
    parser.add_argument("--draft-config-name", default="robotwin_onpolicy_v1a2_draft")
    parser.add_argument("--teacher-config-name", default="robotwin_lingbot_v2a4_teacher")
    parser.add_argument("--save_root", default="/tmp/wanva_single_spec_server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument("--threshold", type=float, default=0.18)
    parser.add_argument("--tau-timesteps", type=float, nargs="+", default=[150.0, 300.0])
    parser.add_argument("--teacher-cache-mode", default="sync", choices=["sync", "lazy_reference", "stale_reference"])
    parser.add_argument("--risk-low", type=float, default=0.25)
    parser.add_argument("--risk-high", type=float, default=0.55)
    parser.add_argument("--risk-verify-mode", default="medium", choices=["off", "medium"])
    parser.add_argument("--risk-delta-ref", type=float, default=0.25)
    parser.add_argument("--risk-mean-delta-ref", type=float, default=0.12)
    parser.add_argument("--risk-jerk-ref", type=float, default=0.18)
    parser.add_argument("--risk-phase-weight", type=float, default=0.25)
    parser.add_argument("--phase-threshold-scale", type=float, default=0.5)
    parser.add_argument("--world-verify-enable", action="store_true")
    parser.add_argument("--world-verify-threshold", type=float, default=0.35)
    parser.add_argument("--world-verify-tau-timesteps", type=float, nargs="+", default=[150.0, 300.0])
    parser.add_argument("--repair-enable", action="store_true")
    parser.add_argument("--repair-lambda", type=float, default=0.75)
    parser.add_argument("--repair-instrument-only", action="store_true")
    parser.add_argument("--svdr-repair-enable", action="store_true")
    parser.add_argument("--svdr-lambda-min", type=float, default=0.05)
    parser.add_argument("--svdr-lambda-max", type=float, default=0.90)
    parser.add_argument("--svdr-motion-ref", type=float, default=3.0)
    parser.add_argument("--svdr-topk-frac", type=float, default=0.10)
    parser.add_argument("--svdr-temperature", type=float, default=1.0)
    parser.add_argument("--specverify-log", default=None)
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
            "server": "wanva_single_spec",
            "draft_config": args.draft_config_name,
            "teacher_config": args.teacher_config_name,
        },
    )
    logging.info("serving single speculative policy on %s:%s", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
