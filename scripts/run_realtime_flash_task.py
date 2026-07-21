#!/usr/bin/env python3
"""Run one RoboTwin task against direct or realtime-flash LingBot serving."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("/mnt/afs/intern/manlichen/ivan/zhoujunl")
CODE = Path(__file__).resolve().parents[1]
ROBOTWIN = ROOT / "Wam_Speed_up" / "RoboTwin"


def runtime_env(
    gpu: int, *, server: bool, deterministic_audit: bool = False
) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["ROBOTWIN_ROOT"] = str(ROBOTWIN)
    paths = []
    if server:
        paths.append(str(ROOT / "env" / "torch29_clean_pkgs"))
    paths.extend([
        str(CODE),
        str(ROBOTWIN),
        str(ROOT / "env" / "curobo-v0.7.8" / "src"),
        str(ROOT / "env" / "python_pkgs"),
        env.get("PYTHONPATH", ""),
    ])
    env["PYTHONPATH"] = ":".join(path for path in paths if path)
    env["PATH"] = (
        f"{ROOT / 'env' / 'ninja-build' / 'extract' / 'usr' / 'bin'}:"
        f"/usr/local/cuda/bin:{env.get('PATH', '')}"
    )
    cuda_lib = "/usr/local/cuda-12.1/targets/x86_64-linux/lib"
    driver_lib = ROOT / "env" / "nvidia-550.90.07-jammy" / "extract" / "usr" / "lib" / "x86_64-linux-gnu"
    env["LD_LIBRARY_PATH"] = (
        f"{driver_lib}:{cuda_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:"
        f"{env.get('LD_LIBRARY_PATH', '')}"
    )
    env["LIBRARY_PATH"] = f"{cuda_lib}:{env.get('LIBRARY_PATH', '')}"
    env["TORCH_EXTENSIONS_DIR"] = str(ROOT / "env" / "torch_extensions")
    env["VK_ICD_FILENAMES"] = str(
        ROOT / "experiments" / "Wam_Speed_up" / "20260709_acp_eval_entrypoints" / "nvidia_icd_abs_egl.json"
    )
    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(
        ROOT / "env" / "python_pkgs" / "sapien" / "vulkan_library" / "10_nvidia.json"
    )
    env["SAPIEN_VULKAN_LIBRARY_PATH"] = str(
        ROOT / "env" / "python_pkgs" / "sapien" / "vulkan_library" / "libvulkan.so.1.3.224"
    )
    env["NVIDIA_DRIVER_CAPABILITIES"] = "all"
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    env["DIFFUSERS_DISABLE_BITSANDBYTES"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONWARNINGS"] = "ignore::UserWarning"
    env["PYTHONUNBUFFERED"] = "1"
    if server and deterministic_audit:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


def wait_for_server(process: subprocess.Popen, port: int, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with rc={process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", int(port)), timeout=1):
                return
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError(f"server was not ready in {timeout}s")


def read_metric(run_root: Path, task: str) -> dict:
    path = run_root / "results" / "stseed-10000" / "metrics" / task / "res.json"
    if not path.exists():
        return {"task": task, "succ_num": 0, "total_num": 0, "succ_rate": 0.0, "status": "missing"}
    data = json.loads(path.read_text())
    return {
        "task": task,
        "succ_num": int(data.get("succ_num", 0)),
        "total_num": int(data.get("total_num", 0)),
        "succ_rate": float(data.get("succ_rate", 0.0)),
        "status": "ok",
    }


def source_summary(path: Path) -> dict:
    counts: dict[str, int] = {}
    latencies: dict[str, list[float]] = {}
    adaptive = {
        "calls": 0,
        "certificates": 0,
        "matches": 0,
        "conflicts": 0,
        "potential_saved_forwards": 0,
        "by_kind": {},
    }
    adaptive_live = {
        "calls": 0,
        "k1_accepts": 0,
        "effective_forwards": 0,
        "requested_forwards": 0,
        "verify_total_sec": [],
        "probe_sec": [],
    }
    verify_profile = {
        "calls": 0,
        "effective_forwards": 0,
        "requested_forwards": 0,
        "verify_total_sec": [],
        "probe_sec": [],
    }
    world_flow = {
        "verify_calls": 0,
        "available_k2": 0,
        "unavailable_k1": 0,
        "missing": 0,
        "endpoint_midpoint_distance_mean": [],
        "cross_tau_half_gap_mean": [],
        "correction_cosine_mean_valid": [],
    }
    draft_evidence = {
        "action_dynamics_calls": 0,
        "video_motion_calls": 0,
        "delayed_video_error_calls": 0,
        "jerk_rms": [],
        "video_global_mean": [],
        "delayed_video_nrmse": [],
    }
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = str(row.get("source", "unknown"))
            counts[source] = counts.get(source, 0) + 1
            if row.get("elapsed_sec") is not None:
                latencies.setdefault(source, []).append(float(row["elapsed_sec"]))
            action_dynamics = row.get("action_dynamics_stats")
            if isinstance(action_dynamics, dict):
                draft_evidence["action_dynamics_calls"] += 1
                if action_dynamics.get("jerk_rms") is not None:
                    value = float(action_dynamics["jerk_rms"])
                    if math.isfinite(value):
                        draft_evidence["jerk_rms"].append(value)
            video_motion = row.get("video_motion_stats")
            if isinstance(video_motion, dict):
                draft_evidence["video_motion_calls"] += 1
                if video_motion.get("global_mean") is not None:
                    value = float(video_motion["global_mean"])
                    if math.isfinite(value):
                        draft_evidence["video_global_mean"].append(value)
            delayed_video = row.get("delayed_video_error")
            if isinstance(delayed_video, dict):
                draft_evidence["delayed_video_error_calls"] += 1
                if delayed_video.get("latent_nrmse") is not None:
                    value = float(delayed_video["latent_nrmse"])
                    if math.isfinite(value):
                        draft_evidence["delayed_video_nrmse"].append(value)
            if row.get("requested_verify_k") is not None:
                world_flow["verify_calls"] += 1
                requested_k = int(row.get("requested_verify_k", 0))
                effective_k = int(row.get("effective_verify_k", 0))
                flow_evidence = row.get("world_flow_evidence")
                if not isinstance(flow_evidence, dict):
                    world_flow["missing"] += 1
                elif (
                    requested_k == 2
                    and effective_k == 2
                    and flow_evidence.get("available")
                ):
                    world_flow["available_k2"] += 1
                    for key in (
                        "endpoint_midpoint_distance_mean",
                        "cross_tau_half_gap_mean",
                        "correction_cosine_mean_valid",
                    ):
                        if flow_evidence.get(key) is not None:
                            value = float(flow_evidence[key])
                            if math.isfinite(value):
                                world_flow[key].append(value)
                elif (
                    requested_k == 2
                    and effective_k == 1
                    and not flow_evidence.get("available")
                    and flow_evidence.get("reason") == "requires_completed_k2"
                ):
                    world_flow["unavailable_k1"] += 1
                else:
                    world_flow["missing"] += 1
                adaptive["calls"] += 1
                kind = row.get("adaptive_k_certificate_kind")
                if kind:
                    kind = str(kind)
                    adaptive["certificates"] += 1
                    adaptive["potential_saved_forwards"] += max(
                        0, int(row["requested_verify_k"]) - 1
                    )
                    bucket = adaptive["by_kind"].setdefault(
                        kind, {"certificates": 0, "matches": 0, "conflicts": 0}
                    )
                    bucket["certificates"] += 1
                    match = row.get("adaptive_k_certificate_matches_full")
                    if match is not None:
                        adaptive["matches"] += int(bool(match))
                        adaptive["conflicts"] += int(not bool(match))
                        bucket["matches"] += int(bool(match))
                        bucket["conflicts"] += int(not bool(match))
                if row.get("adaptive_k_live"):
                    adaptive_live["calls"] += 1
                    adaptive_live["k1_accepts"] += int(
                        bool(row.get("adaptive_k_live_accepted"))
                    )
                    adaptive_live["effective_forwards"] += int(
                        row.get("effective_verify_k", 0)
                    )
                    adaptive_live["requested_forwards"] += int(
                        row.get("requested_verify_k", 0)
                    )
                    if row.get("verify_total_latency_sec") is not None:
                        adaptive_live["verify_total_sec"].append(
                            float(row["verify_total_latency_sec"])
                        )
                    adaptive_live["probe_sec"].extend(
                        float(value)
                        for value in (row.get("verify_probe_latency_sec") or [])
                    )
                if row.get("verify_total_latency_sec") is not None:
                    verify_profile["calls"] += 1
                    verify_profile["effective_forwards"] += int(
                        row.get("effective_verify_k", 0)
                    )
                    verify_profile["requested_forwards"] += int(
                        row.get("requested_verify_k", 0)
                    )
                    verify_profile["verify_total_sec"].append(
                        float(row["verify_total_latency_sec"])
                    )
                    verify_profile["probe_sec"].extend(
                        float(value)
                        for value in (row.get("verify_probe_latency_sec") or [])
                    )
    stats = {}
    for source, values in latencies.items():
        values.sort()
        stats[source] = {
            "count": len(values),
            "p50_sec": values[(len(values) - 1) // 2],
            "p90_sec": values[round((len(values) - 1) * 0.9)],
        }
    action_total = counts.get("teacher_full", 0) + counts.get("draft_flash", 0)
    for key in ("verify_total_sec", "probe_sec"):
        values = sorted(adaptive_live.pop(key))
        adaptive_live[f"{key}_p50"] = (
            values[(len(values) - 1) // 2] if values else None
        )
        adaptive_live[f"{key}_p90"] = (
            values[round((len(values) - 1) * 0.9)] if values else None
        )
    adaptive_live["effective_k_mean"] = (
        adaptive_live["effective_forwards"] / adaptive_live["calls"]
        if adaptive_live["calls"] else None
    )
    for key in ("verify_total_sec", "probe_sec"):
        values = sorted(verify_profile.pop(key))
        verify_profile[f"{key}_p50"] = (
            values[(len(values) - 1) // 2] if values else None
        )
        verify_profile[f"{key}_p90"] = (
            values[round((len(values) - 1) * 0.9)] if values else None
        )
    verify_profile["effective_k_mean"] = (
        verify_profile["effective_forwards"] / verify_profile["calls"]
        if verify_profile["calls"] else None
    )
    for values in (world_flow, draft_evidence):
        for key in tuple(values):
            if not isinstance(values[key], list):
                continue
            samples = values[key]
            values[key] = {
                "count": len(samples),
                "mean": sum(samples) / len(samples) if samples else None,
                "max": max(samples) if samples else None,
            }
    return {
        "counts": counts,
        "latency": stats,
        "teacher_action_rate": counts.get("teacher_full", 0) / action_total if action_total else None,
        "adaptive_k": adaptive,
        "adaptive_k_live": adaptive_live,
        "verify_profile": verify_profile,
        "world_flow": world_flow,
        "draft_evidence": draft_evidence,
    }


def file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("spec", "draft_only", "teacher_only"), default="spec")
    parser.add_argument("--task", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--client-gpu", type=int)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--draft-config-name", default="robotwin_flashwam_step3000_v1a2"
    )
    parser.add_argument(
        "--teacher-config-name", default="robotwin_lingbot_v2a4"
    )
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--tau", type=float, nargs="+", default=(50.0, 100.0))
    parser.add_argument("--pf-interval", type=int, default=2)
    parser.add_argument("--flow-budget-threshold", type=float, default=0.0)
    parser.add_argument("--flow-budget-burst-after", type=int, default=0)
    parser.add_argument("--flow-budget-burst-rounds", type=int, default=0)
    parser.add_argument("--flow-budget-burst-limit", type=int, default=0)
    parser.add_argument("--flow-budget-motion-ceiling", type=float, default=0.0)
    parser.add_argument("--delayed-error-threshold", type=float, default=0.0)
    parser.add_argument("--delayed-error-consecutive", type=int, default=2)
    parser.add_argument("--delayed-error-teacher-rounds", type=int, default=2)
    parser.add_argument("--video-motion-gate-threshold", type=float, default=0.0)
    parser.add_argument("--video-motion-jerk-gate-threshold", type=float, default=0.0)
    parser.add_argument("--video-motion-strict-verify-threshold", type=float, default=0.10)
    parser.add_argument("--video-motion-strict-prefix", type=int, default=16)
    parser.add_argument("--gripper-full-window", type=int, default=1)
    parser.add_argument("--gripper-consensus", action="store_true")
    parser.add_argument(
        "--teacher-gripper-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--server-timeout", type=int, default=600)
    parser.add_argument("--adaptive-k-shadow", action="store_true")
    parser.add_argument("--adaptive-k-live", action="store_true")
    parser.add_argument("--adaptive-k-distance-threshold", type=float, default=0.05)
    parser.add_argument("--profile-verify-latency", action="store_true")
    parser.add_argument("--equivalence-audit", action="store_true")
    parser.add_argument("--deterministic-audit", action="store_true")
    parser.add_argument("--paired-rng", action="store_true")
    parser.add_argument("--policy-seed", type=int)
    parser.add_argument("--scene-manifest-in", type=Path)
    parser.add_argument("--scene-manifest-out", type=Path)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    client_gpu = args.gpu if args.client_gpu is None else args.client_gpu

    for root in (args.run_root, args.result_root):
        (root / "logs").mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    metrics_log = args.run_root / "logs" / f"specverify_{args.task}.jsonl"
    if metrics_log.exists() and metrics_log.stat().st_size:
        raise FileExistsError(
            f"refusing to append to existing policy trace: {metrics_log}"
        )
    server_log_path = args.run_root / "logs" / f"server_{args.task}_g{args.gpu}.log"
    client_log_path = args.run_root / "logs" / f"client_{args.task}_g{client_gpu}.log"
    server_cmd = [
        sys.executable,
        "wan_va/wan_va_single_spec_server.py",
        "--mode", args.mode,
        "--port", str(args.port),
        "--save-root", str(args.run_root / "server" / args.task),
        "--draft-config-name", args.draft_config_name,
        "--teacher-config-name", args.teacher_config_name,
        "--threshold", str(args.threshold),
        "--tau-timesteps", *[str(value) for value in args.tau],
        "--pf-interval", str(args.pf_interval),
        "--flow-budget-threshold", str(args.flow_budget_threshold),
        "--flow-budget-burst-after", str(args.flow_budget_burst_after),
        "--flow-budget-burst-rounds", str(args.flow_budget_burst_rounds),
        "--flow-budget-burst-limit", str(args.flow_budget_burst_limit),
        "--flow-budget-motion-ceiling", str(args.flow_budget_motion_ceiling),
        "--delayed-error-threshold", str(args.delayed_error_threshold),
        "--delayed-error-consecutive", str(args.delayed_error_consecutive),
        "--delayed-error-teacher-rounds", str(args.delayed_error_teacher_rounds),
        "--video-motion-gate-threshold", str(args.video_motion_gate_threshold),
        "--video-motion-jerk-gate-threshold", str(
            args.video_motion_jerk_gate_threshold),
        "--video-motion-strict-verify-threshold", str(
            args.video_motion_strict_verify_threshold),
        "--video-motion-strict-prefix", str(args.video_motion_strict_prefix),
        "--gripper-full-window", str(args.gripper_full_window),
        "--adaptive-k-distance-threshold", str(
            args.adaptive_k_distance_threshold),
        "--log-path", str(metrics_log),
    ]
    if args.policy_seed is not None:
        server_cmd.extend(["--seed", str(args.policy_seed)])
    if args.gripper_consensus:
        server_cmd.append("--gripper-consensus")
    if not args.teacher_gripper_fallback:
        server_cmd.append("--no-teacher-gripper-fallback")
    if args.adaptive_k_shadow:
        server_cmd.append("--adaptive-k-shadow")
    if args.adaptive_k_live:
        server_cmd.append("--adaptive-k-live")
    if args.profile_verify_latency:
        server_cmd.append("--profile-verify-latency")
    if args.equivalence_audit:
        server_cmd.append("--equivalence-audit")
    if args.deterministic_audit:
        server_cmd.append("--deterministic-audit")
    client_cmd = [
        sys.executable,
        "-m", "evaluation.robotwin.eval_polict_client_openpi",
        "--config", "policy/ACT/deploy_policy.yml",
        "--overrides",
        "--task_name", args.task,
        "--task_config", "demo_clean",
        "--train_config_name", "0",
        "--model_name", "0",
        "--ckpt_setting", f"realtime-flash-{args.mode}",
        "--seed", "0",
        "--policy_name", "ACT",
        "--save_root", str(args.run_root / "results"),
        "--video_guidance_scale", "5",
        "--action_guidance_scale", "1",
        "--test_num", str(args.test_num),
        "--port", str(args.port),
    ]
    if args.scene_manifest_in:
        client_cmd.extend(["--scene_manifest_in", str(args.scene_manifest_in)])
    if args.scene_manifest_out:
        client_cmd.extend(["--scene_manifest_out", str(args.scene_manifest_out)])
    if args.manifest_only:
        client_cmd.extend(["--manifest_only", "True"])
    if args.paired_rng:
        client_cmd.extend(["--paired_rng", "True"])
    command = {
        "server": server_cmd,
        "client": client_cmd,
        "server_gpu": args.gpu,
        "client_gpu": client_gpu,
        "started_at": started,
    }
    (args.result_root / f"command_{args.task}.json").write_text(json.dumps(command, indent=2))

    server_log = None
    server = None
    if not args.manifest_only:
        server_log = server_log_path.open("a", buffering=1)
        server = subprocess.Popen(
            server_cmd,
            cwd=CODE,
            env=runtime_env(
                args.gpu,
                server=True,
                deterministic_audit=args.deterministic_audit,
            ),
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
    client_rc = None
    try:
        if server is not None:
            wait_for_server(server, args.port, args.server_timeout)
        with client_log_path.open("a", buffering=1) as client_log:
            client = subprocess.run(
                client_cmd,
                cwd=CODE,
                env=runtime_env(client_gpu, server=False),
                stdout=client_log,
                stderr=subprocess.STDOUT,
            )
        client_rc = client.returncode
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        if server_log is not None:
            server_log.close()

    metric = read_metric(args.run_root, args.task)
    summary = {
        "mode": args.mode,
        "task": args.task,
        "gpu": args.gpu,
        "client_gpu": client_gpu,
        "threshold": args.threshold,
        "draft_config_name": args.draft_config_name,
        "teacher_config_name": args.teacher_config_name,
        "tau": args.tau,
        "pf_interval": args.pf_interval,
        "flow_budget_threshold": args.flow_budget_threshold,
        "flow_budget_burst_after": args.flow_budget_burst_after,
        "flow_budget_burst_rounds": args.flow_budget_burst_rounds,
        "flow_budget_burst_limit": args.flow_budget_burst_limit,
        "flow_budget_motion_ceiling": args.flow_budget_motion_ceiling,
        "delayed_error_threshold": args.delayed_error_threshold,
        "delayed_error_consecutive": args.delayed_error_consecutive,
        "delayed_error_teacher_rounds": args.delayed_error_teacher_rounds,
        "video_motion_gate_threshold": args.video_motion_gate_threshold,
        "video_motion_jerk_gate_threshold": args.video_motion_jerk_gate_threshold,
        "video_motion_strict_verify_threshold": (
            args.video_motion_strict_verify_threshold
        ),
        "video_motion_strict_prefix": args.video_motion_strict_prefix,
        "gripper_full_window": args.gripper_full_window,
        "gripper_consensus": args.gripper_consensus,
        "teacher_gripper_fallback": args.teacher_gripper_fallback,
        "adaptive_k_shadow": args.adaptive_k_shadow,
        "adaptive_k_live": args.adaptive_k_live,
        "adaptive_k_distance_threshold": args.adaptive_k_distance_threshold,
        "profile_verify_latency": args.profile_verify_latency,
        "equivalence_audit": args.equivalence_audit,
        "deterministic_audit": args.deterministic_audit,
        "paired_rng": args.paired_rng,
        "policy_seed": args.policy_seed,
        "scene_manifest_in": (
            str(args.scene_manifest_in) if args.scene_manifest_in else None
        ),
        "scene_manifest_out": (
            str(args.scene_manifest_out) if args.scene_manifest_out else None
        ),
        "scene_manifest_sha256": file_sha256(
            args.scene_manifest_in or args.scene_manifest_out
        ),
        "manifest_only": args.manifest_only,
        "started_at": started,
        "ended_at": datetime.now().isoformat(timespec="seconds"),
        "client_rc": client_rc,
        "metric": metric,
        "policy": source_summary(metrics_log),
    }
    for root in (args.run_root, args.result_root):
        (root / f"summary_{args.task}.json").write_text(json.dumps(summary, indent=2))
    for path in (server_log_path, client_log_path, metrics_log):
        if path.exists():
            shutil.copy2(path, args.result_root / "logs" / path.name)
    print(json.dumps(summary, indent=2), flush=True)
    if client_rc != 0 or (
        not args.manifest_only and metric["total_num"] < args.test_num
    ):
        raise SystemExit(client_rc or 2)


if __name__ == "__main__":
    main()
