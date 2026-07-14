#!/usr/bin/env python3
"""Run one RoboTwin task against direct or realtime-flash LingBot serving."""

from __future__ import annotations

import argparse
import json
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


def runtime_env(gpu: int, *, server: bool) -> dict[str, str]:
    env = os.environ.copy()
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
    repair = {
        "eligible": 0,
        "attempted": 0,
        "executed": 0,
        "action_only_forwards": 0,
        "kinds": {},
    }
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = str(row.get("source", "unknown"))
            counts[source] = counts.get(source, 0) + 1
            repair["eligible"] += int(bool(row.get("repair_eligible", False)))
            repair["attempted"] += int(bool(row.get("repair_attempted", False)))
            repair["executed"] += int(bool(row.get("repair_executed", False)))
            repair["action_only_forwards"] += int(
                row.get("repair_action_only_forwards", 0) or 0
            )
            if row.get("repair_kind"):
                kind = str(row["repair_kind"])
                repair["kinds"][kind] = repair["kinds"].get(kind, 0) + 1
            if row.get("elapsed_sec") is not None:
                latencies.setdefault(source, []).append(float(row["elapsed_sec"]))
    stats = {}
    for source, values in latencies.items():
        values.sort()
        stats[source] = {
            "count": len(values),
            "p50_sec": values[(len(values) - 1) // 2],
            "p90_sec": values[round((len(values) - 1) * 0.9)],
        }
    action_total = counts.get("teacher_full", 0) + counts.get("draft_flash", 0)
    return {
        "counts": counts,
        "latency": stats,
        "teacher_action_rate": counts.get("teacher_full", 0) / action_total if action_total else None,
        "repair": repair,
    }


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
    parser.add_argument("--gripper-full-window", type=int, default=1)
    parser.add_argument("--gripper-consensus", action="store_true")
    parser.add_argument("--gripper-repair", action="store_true")
    parser.add_argument("--gripper-repair-tau", type=float, default=75.0)
    parser.add_argument("--zero-prefix-repair", action="store_true")
    parser.add_argument("--repair-max-per-episode", type=int, default=2)
    parser.add_argument("--repair-shadow", action="store_true")
    parser.add_argument("--repair-strength", type=float, default=0.5)
    parser.add_argument("--repair-max-step-rms", type=float, default=0.15)
    parser.add_argument("--repair-prefix-len", type=int, default=16)
    parser.add_argument(
        "--teacher-gripper-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--server-timeout", type=int, default=600)
    args = parser.parse_args()
    client_gpu = args.gpu if args.client_gpu is None else args.client_gpu

    for root in (args.run_root, args.result_root):
        (root / "logs").mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    metrics_log = args.run_root / "logs" / f"specverify_{args.task}.jsonl"
    server_log_path = args.run_root / "logs" / f"server_{args.task}_g{args.gpu}.log"
    client_log_path = args.run_root / "logs" / f"client_{args.task}_g{client_gpu}.log"
    server_cmd = [
        sys.executable,
        "wan_va/wan_va_single_spec_server.py",
        "--mode", args.mode,
        "--port", str(args.port),
        "--save-root", str(args.run_root / "server" / args.task),
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
        "--gripper-full-window", str(args.gripper_full_window),
        "--log-path", str(metrics_log),
    ]
    if args.gripper_consensus:
        server_cmd.append("--gripper-consensus")
    if args.gripper_repair:
        server_cmd.append("--gripper-repair")
    if args.zero_prefix_repair:
        server_cmd.append("--zero-prefix-repair")
    server_cmd.extend([
        "--gripper-repair-tau", str(args.gripper_repair_tau),
        "--repair-max-per-episode", str(args.repair_max_per_episode),
    ])
    if args.repair_shadow:
        server_cmd.append("--repair-shadow")
    server_cmd.extend([
        "--repair-strength", str(args.repair_strength),
        "--repair-max-step-rms", str(args.repair_max_step_rms),
        "--repair-prefix-len", str(args.repair_prefix_len),
    ])
    if not args.teacher_gripper_fallback:
        server_cmd.append("--no-teacher-gripper-fallback")
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
    command = {
        "server": server_cmd,
        "client": client_cmd,
        "server_gpu": args.gpu,
        "client_gpu": client_gpu,
        "started_at": started,
    }
    (args.result_root / f"command_{args.task}.json").write_text(json.dumps(command, indent=2))

    server_log = server_log_path.open("a", buffering=1)
    server = subprocess.Popen(
        server_cmd,
        cwd=CODE,
        env=runtime_env(args.gpu, server=True),
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )
    client_rc = None
    try:
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
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        server_log.close()

    metric = read_metric(args.run_root, args.task)
    summary = {
        "mode": args.mode,
        "task": args.task,
        "gpu": args.gpu,
        "client_gpu": client_gpu,
        "threshold": args.threshold,
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
        "gripper_full_window": args.gripper_full_window,
        "gripper_consensus": args.gripper_consensus,
        "gripper_repair": args.gripper_repair,
        "gripper_repair_tau": args.gripper_repair_tau,
        "zero_prefix_repair": args.zero_prefix_repair,
        "repair_max_per_episode": args.repair_max_per_episode,
        "repair_shadow": args.repair_shadow,
        "repair_strength": args.repair_strength,
        "repair_max_step_rms": args.repair_max_step_rms,
        "repair_prefix_len": args.repair_prefix_len,
        "teacher_gripper_fallback": args.teacher_gripper_fallback,
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
    if client_rc != 0 or metric["total_num"] < args.test_num:
        raise SystemExit(client_rc or 2)


if __name__ == "__main__":
    main()
