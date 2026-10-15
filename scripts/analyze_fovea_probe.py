"""Summarize fovea resolution-sensitivity probe logs.

Usage:
  python scripts/analyze_fovea_probe.py results/fovea_probe/*/*.jsonl [--baseline all-all-x1-either]

Each log line is written by evaluation/robotwin/fovea_probe.py. The baseline is
the factor-1 run (no degradation, same wrapper and logging). Episodes are paired
across conditions by (task, seed); RoboTwin evaluation walks the same seed
sequence when started from the same --seed, so paired differences are valid.
"""

import argparse
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.robotwin.fovea_probe import load_records  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def mcnemar_exact_p(b, c):
    """Two-sided exact McNemar p-value from discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def task_of(path):
    return os.path.basename(os.path.dirname(path)) or "task"


def summarize(paths, baseline):
    episodes = defaultdict(dict)  # condition -> {(task, seed or index): success}
    chunks = defaultdict(lambda: {"n": 0, "fine": 0, "switch": 0, "degraded": 0, "speeds": []})
    for path in paths:
        task = task_of(path)
        for r in load_records([path]):
            cond = r["condition"]
            if r["event"] == "episode":
                key = (task, r.get("seed") if r.get("seed") is not None else r["episode"])
                episodes[cond][key] = bool(r["success"])
            elif r["event"] == "chunk":
                c = chunks[cond]
                c["n"] += 1
                c["fine"] += int(r["fine"])
                c["switch"] += int(r["gripper_switch"])
                c["degraded"] += int(r["degraded"])
                if r.get("speed") is not None and math.isfinite(r["speed"]):
                    c["speeds"].append(r["speed"])

    base = episodes.get(baseline, {})
    rows = []
    for cond in sorted(episodes):
        eps = episodes[cond]
        n, k = len(eps), sum(eps.values())
        lo, hi = wilson(k, n)
        row = {"condition": cond, "n": n, "success": k, "rate": k / n if n else float("nan"),
               "ci_lo": lo, "ci_hi": hi}
        shared = sorted(set(eps) & set(base))
        if cond != baseline and shared:
            b = sum(1 for s in shared if base[s] and not eps[s])  # broken by degradation
            c = sum(1 for s in shared if eps[s] and not base[s])  # fixed by degradation
            row.update({"paired_n": len(shared),
                        "delta": (sum(eps[s] for s in shared) - sum(base[s] for s in shared)) / len(shared),
                        "broken": b, "fixed": c, "mcnemar_p": mcnemar_exact_p(b, c)})
        ch = chunks.get(cond)
        if ch and ch["n"]:
            speeds = sorted(ch["speeds"])
            row.update({"chunks": ch["n"], "fine_frac": ch["fine"] / ch["n"],
                        "switch_frac": ch["switch"] / ch["n"], "degraded_frac": ch["degraded"] / ch["n"],
                        "speed_p25": speeds[len(speeds) // 4] if speeds else float("nan"),
                        "speed_p50": speeds[len(speeds) // 2] if speeds else float("nan")})
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--baseline", default="all-all-x1-either")
    args = ap.parse_args()
    rows = summarize(args.logs, args.baseline)
    cols = ["condition", "n", "rate", "ci_lo", "ci_hi", "paired_n", "delta", "broken", "fixed",
            "mcnemar_p", "degraded_frac", "fine_frac", "switch_frac", "speed_p25", "speed_p50"]
    print("\t".join(cols))
    for r in rows:
        out = []
        for c in cols:
            v = r.get(c, "")
            out.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        print("\t".join(out))


if __name__ == "__main__":
    main()
