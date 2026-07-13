#!/usr/bin/env python3
"""Audit draft-motion labels and compare pre-execution motion scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable


HIGH_ERROR_THRESHOLD = 0.55
BASELINE_GATE_THRESHOLD = 1.2
STRONG_TAIL_DISTANCE = 0.05


def _task_from_path(path: Path) -> str:
    name = path.stem
    return name.removeprefix("specverify_")


def audit_log(
    path: Path,
    run_name: str | None = None,
    verify_threshold: float | None = None,
) -> tuple[list[dict], Counter]:
    """Pair each executed draft with its acknowledged delayed-error label."""

    task = _task_from_path(path)
    run_name = run_name or (
        path.parent.parent.name if path.parent.name == "logs" else path.parent.name
    )
    episode = -1
    pending: dict[int, dict] = {}
    rows: list[dict] = []
    audit = Counter()

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            source = record.get("source")
            round_id = record.get("round_id")

            if source == "reset":
                for proposal in pending.values():
                    audit["excluded_missing_cache_update"] += 1
                    proposal["exclusion_reason"] = "episode_reset_before_cache_update"
                pending.clear()
                episode += 1
                continue

            motion = record.get("video_motion_stats")
            if motion is not None:
                if source != "draft_flash" or int(record.get("accepted_prefix", 0)) <= 0:
                    audit["excluded_unexecuted_motion_proposal"] += 1
                    continue
                if round_id in pending:
                    audit["excluded_overwritten_proposal"] += 1
                proposal = {
                    "run": run_name,
                    "task": task,
                    "episode": episode,
                    "round_id": round_id,
                    "proposal_line": line_number,
                    "accepted_prefix": int(record["accepted_prefix"]),
                    "accepted_prefix_before_motion_cap": int(
                        record.get(
                            "accepted_prefix_before_motion_cap",
                            record["accepted_prefix"],
                        )
                    ),
                    "fallback_reason": record.get("fallback_reason"),
                    "verify_threshold": record.get(
                        "verify_threshold", verify_threshold
                    ),
                    "prefix_by_tau": record.get("prefix_by_tau"),
                    "verify_distances": record.get("verify_distances"),
                    "gripper_consensus_failure_index": record.get(
                        "gripper_consensus_failure_index"
                    ),
                    **motion,
                }
                _add_verifier_margin(proposal, audit)
                pending[round_id] = proposal
                audit["executed_motion_proposal"] += 1

            if source != "flash_cache_update":
                continue
            proposal = pending.pop(round_id, None)
            if proposal is None:
                audit["excluded_unmatched_flash_cache_update"] += 1
                continue
            delayed = record.get("delayed_video_error")
            if delayed is None:
                audit["excluded_missing_delayed_error"] += 1
                continue
            expected_frames = math.ceil(proposal["accepted_prefix"] / 16)
            actual_frames = int(record.get("cache_frame_count", -1))
            if actual_frames != expected_frames:
                audit["excluded_frame_alignment_mismatch"] += 1
                continue

            proposal.update(
                cache_line=line_number,
                cache_frame_count=actual_frames,
                latent_nrmse=float(delayed["latent_nrmse"]),
                high_delayed_error=float(delayed["latent_nrmse"])
                > HIGH_ERROR_THRESHOLD,
            )
            rows.append(proposal)
            audit["included_pair"] += 1

    audit["excluded_missing_cache_update"] += len(pending)
    return rows, audit


def _add_verifier_margin(proposal: dict, audit: Counter) -> None:
    """Attach full-prefix tail margin without changing pair eligibility."""

    distances = proposal.get("verify_distances")
    threshold = proposal.get("verify_threshold")
    if distances is None or threshold is None:
        audit["missing_verifier_margin"] += 1
        return
    if not isinstance(distances, list) or not distances:
        audit["malformed_verifier_margin"] += 1
        return
    try:
        threshold = float(threshold)
        per_tau = [
            [float(value) for frame in tau for value in frame]
            for tau in distances
        ]
    except (TypeError, ValueError):
        audit["malformed_verifier_margin"] += 1
        return
    if not math.isfinite(threshold) or threshold <= 0 or any(
        not math.isfinite(value) for row in per_tau for value in row
    ):
        audit["malformed_verifier_margin"] += 1
        return
    if any(len(row) < 32 for row in per_tau):
        audit["short_verifier_horizon"] += 1
        return
    tail_max = max(max(row[16:32]) for row in per_tau)
    proposal.update(
        verifier_tail_max=tail_max,
        verifier_tail_margin=1.0 - tail_max / threshold,
        verifier_tail_strong=tail_max <= STRONG_TAIL_DISTANCE,
    )
    audit["included_verifier_margin"] += 1


def roc_auc(scores: list[float], labels: list[bool]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = sum(
        (positive > negative) + 0.5 * (positive == negative)
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def average_precision(scores: list[float], labels: list[bool]) -> float | None:
    positive_count = sum(labels)
    if not positive_count:
        return None
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    hits = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, 1):
        if labels[index]:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positive_count


def _matched_budget_metrics(
    rows: list[dict], score: Callable[[dict], float], trigger_count: int
) -> dict:
    selected = sorted(rows, key=score, reverse=True)[:trigger_count]
    true_positives = sum(row["high_delayed_error"] for row in selected)
    positives = sum(row["high_delayed_error"] for row in rows)
    return {
        "trigger_count": len(selected),
        "trigger_rate": len(selected) / len(rows) if rows else 0.0,
        "true_positives": true_positives,
        "precision": true_positives / len(selected) if selected else None,
        "recall": true_positives / positives if positives else None,
    }


def _score_functions(rows: list[dict]) -> dict[str, Callable[[dict], float]]:
    scores: dict[str, Callable[[dict], float]] = {
        "global_mean": lambda row: float(row["global_mean"]),
        "median": lambda row: float(row["median"]),
    }
    if rows and all("motion_v2_score" in row for row in rows):
        scores["motion_v2"] = lambda row: float(row["motion_v2_score"])
    return scores


def compare_scores(rows: list[dict]) -> dict:
    labels = [bool(row["high_delayed_error"]) for row in rows]
    baseline_trigger_count = sum(
        float(row["global_mean"]) >= BASELINE_GATE_THRESHOLD for row in rows
    )
    scores = _score_functions(rows)
    overall = {}
    by_task = {}
    leave_one_task_out = {}
    task_balanced_budget = {}
    for name, score in scores.items():
        values = [score(row) for row in rows]
        overall[name] = {
            "roc_auc": roc_auc(values, labels),
            "average_precision": average_precision(values, labels),
            "matched_budget": _matched_budget_metrics(
                rows, score, baseline_trigger_count
            ),
        }
        by_task[name] = {}
        for task in sorted({row["task"] for row in rows}):
            heldout = [row for row in rows if row["task"] == task]
            heldout_labels = [bool(row["high_delayed_error"]) for row in heldout]
            heldout_values = [score(row) for row in heldout]
            by_task[name][task] = {
                "count": len(heldout),
                "positive_count": sum(heldout_labels),
                "roc_auc": roc_auc(heldout_values, heldout_labels),
                "average_precision": average_precision(
                    heldout_values, heldout_labels
                ),
            }
        task_average_precisions = [
            result["average_precision"]
            for result in by_task[name].values()
            if result["average_precision"] is not None
        ]
        overall[name]["macro_task_average_precision"] = (
            sum(task_average_precisions) / len(task_average_precisions)
            if task_average_precisions
            else None
        )
        fold_rows = []
        for task in sorted({row["task"] for row in rows}):
            train = [row for row in rows if row["task"] != task]
            heldout = [row for row in rows if row["task"] == task]
            train_trigger_count = max(
                1, round(len(train) * baseline_trigger_count / len(rows))
            )
            train_scores = sorted((score(row) for row in train), reverse=True)
            threshold = train_scores[train_trigger_count - 1]
            selected = [score(row) >= threshold for row in heldout]
            trigger_count = sum(selected)
            true_positives = sum(
                flag and row["high_delayed_error"]
                for flag, row in zip(selected, heldout)
            )
            positives = sum(row["high_delayed_error"] for row in heldout)
            fold_rows.append(
                {
                    "task": task,
                    "train_count": len(train),
                    "heldout_count": len(heldout),
                    "threshold": threshold,
                    "trigger_count": trigger_count,
                    "trigger_rate": trigger_count / len(heldout),
                    "true_positives": true_positives,
                    "precision": true_positives / trigger_count
                    if trigger_count
                    else None,
                    "recall": true_positives / positives if positives else None,
                }
            )
        total_triggers = sum(fold["trigger_count"] for fold in fold_rows)
        total_true_positives = sum(
            fold["true_positives"] for fold in fold_rows
        )
        total_positives = sum(row["high_delayed_error"] for row in rows)
        leave_one_task_out[name] = {
            "folds": fold_rows,
            "trigger_count": total_triggers,
            "trigger_rate": total_triggers / len(rows),
            "true_positives": total_true_positives,
            "precision": total_true_positives / total_triggers
            if total_triggers
            else None,
            "recall": total_true_positives / total_positives
            if total_positives
            else None,
        }
        task_budget_rows = []
        for task in sorted({row["task"] for row in rows}):
            heldout = [row for row in rows if row["task"] == task]
            task_trigger_count = max(
                1, round(len(heldout) * baseline_trigger_count / len(rows))
            )
            metrics = _matched_budget_metrics(
                heldout, score, task_trigger_count
            )
            metrics["task"] = task
            task_budget_rows.append(metrics)
        task_triggers = sum(row["trigger_count"] for row in task_budget_rows)
        task_true_positives = sum(
            row["true_positives"] for row in task_budget_rows
        )
        task_balanced_budget[name] = {
            "folds": task_budget_rows,
            "trigger_count": task_triggers,
            "trigger_rate": task_triggers / len(rows),
            "true_positives": task_true_positives,
            "precision": task_true_positives / task_triggers,
            "recall": task_true_positives / total_positives,
        }
    return {
        "pair_count": len(rows),
        "positive_count": sum(labels),
        "high_error_threshold": HIGH_ERROR_THRESHOLD,
        "baseline_gate_threshold": BASELINE_GATE_THRESHOLD,
        "baseline_trigger_count": baseline_trigger_count,
        "overall": overall,
        "within_task_ranking": by_task,
        "task_balanced_matched_budget": task_balanced_budget,
        "leave_one_task_out_threshold": leave_one_task_out,
    }


def write_pairs(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--verify-threshold",
        type=float,
        default=None,
        help="explicit fallback for legacy logs that predate threshold telemetry",
    )
    args = parser.parse_args(argv)

    rows: list[dict] = []
    audit = Counter()
    per_log = []
    for path in args.logs:
        log_rows, log_audit = audit_log(
            path, verify_threshold=args.verify_threshold
        )
        rows.extend(log_rows)
        audit.update(log_audit)
        per_log.append({"path": str(path), "audit": dict(log_audit)})

    if not rows:
        raise RuntimeError("no valid executed-draft motion/error pairs")

    identities = {
        (row["run"], row["task"], row["episode"], row["round_id"])
        for row in rows
    }
    if len(identities) != len(rows):
        raise RuntimeError("duplicate run/task/episode/round motion pairs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_pairs(args.output_dir / "motion_pairs.csv", rows)
    report = {
        "audit": dict(audit),
        "per_log": per_log,
        "comparison": compare_scores(rows),
    }
    (args.output_dir / "motion_score_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
