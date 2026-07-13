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
TARGET_TRIGGER_RATE = 0.067
STRONG_TAIL_DISTANCE = 0.05
ACTION_PER_FRAME = 16
FULL_DRAFT_PREFIX = 32
ONE_SIDED_ALPHA = 0.05


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
    previous_saliency: dict[str, float] | None = None

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
                previous_saliency = None
                episode += 1
                continue

            if source in {"teacher_full", "full_cache_update", "replan"}:
                previous_saliency = None

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
                previous_saliency = None
                continue
            expected_frames = math.ceil(proposal["accepted_prefix"] / 16)
            actual_frames = int(record.get("cache_frame_count", -1))
            if actual_frames != expected_frames:
                audit["excluded_frame_alignment_mismatch"] += 1
                previous_saliency = None
                continue

            proposal.update(
                cache_line=line_number,
                cache_frame_count=actual_frames,
                latent_nrmse=float(delayed["latent_nrmse"]),
                high_delayed_error=float(delayed["latent_nrmse"])
                > HIGH_ERROR_THRESHOLD,
            )
            current_saliency = _region_saliency(proposal)
            if current_saliency is None:
                previous_saliency = None
                audit["missing_motion_v3_saliency"] += 1
            else:
                if (
                    previous_saliency is not None
                    and current_saliency.keys() == previous_saliency.keys()
                ):
                    proposal["motion_v3_history_innovation"] = max(
                        abs(current_saliency[name] - previous_saliency[name])
                        / (
                            abs(current_saliency[name])
                            + abs(previous_saliency[name])
                            + 1e-8
                        )
                        for name in current_saliency
                    )
                    audit["included_motion_v3_history"] += 1
                elif previous_saliency is not None:
                    audit["motion_v3_region_mismatch"] += 1
                previous_saliency = current_saliency
            rows.append(proposal)
            audit["included_pair"] += 1

    audit["excluded_missing_cache_update"] += len(pending)
    return rows, audit


def audit_second_probe_log(
    path: Path,
    run_name: str | None = None,
) -> tuple[list[dict], Counter]:
    """Collect every standard K=2 proposal, including rejected proposals."""

    task = _task_from_path(path)
    run_name = run_name or (
        path.parent.parent.name if path.parent.name == "logs" else path.parent.name
    )
    episode = -1
    rows: list[dict] = []
    audit = Counter()

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            if record.get("source") == "reset":
                episode += 1
                continue
            prefixes = record.get("prefix_by_tau")
            agreements = record.get("gripper_phase_agreement_by_tau")
            taus = record.get("active_tau_timesteps", record.get("tau_timesteps"))
            distances = record.get("verify_distances")
            if taus is None:
                if record.get("video_motion_stats") is not None:
                    audit["excluded_missing_k2_telemetry"] += 1
                continue
            try:
                taus = [float(value) for value in taus]
            except (TypeError, ValueError):
                audit["excluded_malformed_k2_telemetry"] += 1
                continue
            if taus != [50.0, 100.0]:
                audit["excluded_nonstandard_k2_probe"] += 1
                continue
            audit["observed_standard_k2_proposal"] += 1
            motion = record.get("video_motion_stats")
            if motion is None:
                audit["excluded_missing_motion"] += 1
                continue
            if prefixes is None or agreements is None or distances is None:
                audit["excluded_missing_k2_telemetry"] += 1
                continue
            try:
                prefixes = [int(value) for value in prefixes]
                agreements = [float(value) for value in agreements]
                all_distances = [
                    [float(value) for frame in tau_distances for value in frame]
                    for tau_distances in distances
                ]
            except (IndexError, TypeError, ValueError):
                audit["excluded_malformed_k2_telemetry"] += 1
                continue
            if len(prefixes) != 2 or len(agreements) != 2 or len(all_distances) != 2:
                audit["excluded_malformed_k2_telemetry"] += 1
                continue
            if any(prefix < 0 or prefix > FULL_DRAFT_PREFIX for prefix in prefixes):
                audit["excluded_malformed_k2_telemetry"] += 1
                continue
            if any(not values for values in all_distances) or any(
                not math.isfinite(value)
                for value in (*agreements, *(v for values in all_distances for v in values))
            ):
                audit["excluded_malformed_k2_telemetry"] += 1
                continue
            try:
                baseline_motion = [
                    float(motion[key]) for key in ("global_mean", "median")
                ]
            except (KeyError, TypeError, ValueError):
                audit["excluded_malformed_motion"] += 1
                continue
            if not all(math.isfinite(value) for value in baseline_motion):
                audit["excluded_malformed_motion"] += 1
                continue

            draft_switch = record.get("draft_gripper_switch_index")
            switches = record.get("gripper_switch_indices_by_tau")
            if switches is None or not isinstance(switches, list) or len(switches) != 2:
                audit["excluded_missing_phase_switch_telemetry"] += 1
                continue

            quantized = [
                max(0, prefix) // ACTION_PER_FRAME * ACTION_PER_FRAME
                for prefix in prefixes
            ]
            first_phase_stable = math.isclose(
                agreements[0], 1.0, rel_tol=0.0, abs_tol=1e-8
            )
            second_phase_stable = math.isclose(
                agreements[1], 1.0, rel_tol=0.0, abs_tol=1e-8
            )
            second_restricts_prefix = quantized[1] < quantized[0]
            second_restricts_phase = first_phase_stable and not second_phase_stable
            rows.append(
                {
                    "run": run_name,
                    "task": task,
                    "episode": episode,
                    "round_id": record.get("round_id"),
                    "proposal_line": line_number,
                    "tau50_prefix": quantized[0],
                    "tau100_prefix": quantized[1],
                    "tau50_max_residual": max(all_distances[0]),
                    "tau50_gripper_stable": first_phase_stable,
                    "draft_gripper_switch_index": draft_switch,
                    "tau50_gripper_switch_index": switches[0],
                    "tau50_no_phase_switch": (
                        draft_switch is None and switches[0] is None
                    ),
                    "second_probe_restricts_prefix": second_restricts_prefix,
                    "second_probe_restricts_phase": second_restricts_phase,
                    "second_probe_restricts": (
                        second_restricts_prefix or second_restricts_phase
                    ),
                    **motion,
                }
            )
            audit["included_k2_proposal"] += 1
    return rows, audit


def _region_saliency(proposal: dict) -> dict[str, float] | None:
    regions = proposal.get("motion_v2_regions")
    if not isinstance(regions, dict) or not regions:
        return None
    try:
        values = {
            str(name): float(stats["saliency_weighted"])
            for name, stats in regions.items()
        }
    except (KeyError, TypeError, ValueError):
        return None
    if any(not math.isfinite(value) for value in values.values()):
        return None
    return values


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
    groups: dict[float, list[bool]] = {}
    for score, label in zip(scores, labels):
        groups.setdefault(score, []).append(label)
    true_positives = 0
    false_positives = 0
    result = 0.0
    for score in sorted(groups, reverse=True):
        group = groups[score]
        group_positives = sum(group)
        true_positives += group_positives
        false_positives += len(group) - group_positives
        precision = true_positives / (true_positives + false_positives)
        result += group_positives / positive_count * precision
    return result


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
    if rows and all("motion_v3_saliency_score" in row for row in rows):
        scores["motion_v3_saliency"] = lambda row: float(
            row["motion_v3_saliency_score"]
        )
    return scores


def _binomial_cdf(failures: int, count: int, probability: float) -> float:
    return sum(
        math.comb(count, value)
        * probability ** value
        * (1.0 - probability) ** (count - value)
        for value in range(failures + 1)
    )


def _clopper_pearson_upper(failures: int, count: int) -> float | None:
    if count <= 0:
        return None
    if failures >= count:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if _binomial_cdf(failures, count, midpoint) > ONE_SIDED_ALPHA:
            low = midpoint
        else:
            high = midpoint
    return high


def _zero_miss_threshold(
    rows: list[dict], score: Callable[[dict], float]
) -> float | None:
    if not rows:
        return None
    risky_scores = [score(row) for row in rows if row["second_probe_restricts"]]
    if not risky_scores:
        return None
    return math.nextafter(min(risky_scores), -math.inf)


def compare_second_probe_scores(rows: list[dict]) -> dict:
    """Evaluate motion only as a veto on the strict tau-50 fast path."""

    strict = [
        row for row in rows
        if row["tau50_prefix"] >= FULL_DRAFT_PREFIX
        and row["tau50_max_residual"] < STRONG_TAIL_DISTANCE
        and row["tau50_gripper_stable"]
        and row["tau50_no_phase_switch"]
    ]
    scores = _score_functions(strict)
    labels = [bool(row["second_probe_restricts"]) for row in strict]
    result = {}
    tasks = sorted({row["task"] for row in strict})
    for name, score in scores.items():
        folds = []
        selected_total = 0
        failures_total = 0
        selected_episodes = set()
        failure_episodes = set()
        for task in tasks:
            train = [row for row in strict if row["task"] != task]
            heldout = [row for row in strict if row["task"] == task]
            threshold = _zero_miss_threshold(train, score)
            selected = (
                [row for row in heldout if score(row) <= threshold]
                if threshold is not None else []
            )
            failures = sum(row["second_probe_restricts"] for row in selected)
            heldout_selected_episodes = {
                (row["run"], row["task"], row["episode"]) for row in selected
            }
            heldout_failure_episodes = {
                (row["run"], row["task"], row["episode"])
                for row in selected if row["second_probe_restricts"]
            }
            selected_total += len(selected)
            failures_total += failures
            selected_episodes.update(heldout_selected_episodes)
            failure_episodes.update(heldout_failure_episodes)
            folds.append(
                {
                    "task": task,
                    "train_count": len(train),
                    "heldout_count": len(heldout),
                    "threshold": threshold,
                    "selected_count": len(selected),
                    "coverage": len(selected) / len(heldout) if heldout else None,
                    "failures": failures,
                    "false_safe_rate": failures / len(selected) if selected else None,
                    "false_safe_upper_95": _clopper_pearson_upper(
                        failures, len(selected)
                    ),
                    "selected_episode_count": len(heldout_selected_episodes),
                    "failure_episode_count": len(heldout_failure_episodes),
                    "episode_false_safe_upper_95": _clopper_pearson_upper(
                        len(heldout_failure_episodes),
                        len(heldout_selected_episodes),
                    ),
                    "insufficient_positive_support": threshold is None,
                }
            )
        result[name] = {
            "roc_auc": roc_auc([score(row) for row in strict], labels),
            "average_precision": average_precision(
                [score(row) for row in strict], labels
            ),
            "folds": folds,
            "selected_count": selected_total,
            "coverage": selected_total / len(strict) if strict else 0.0,
            "failures": failures_total,
            "false_safe_rate": (
                failures_total / selected_total if selected_total else None
            ),
            "false_safe_upper_95": _clopper_pearson_upper(
                failures_total, selected_total
            ),
            "selected_episode_count": len(selected_episodes),
            "failure_episode_count": len(failure_episodes),
            "episode_false_safe_rate": (
                len(failure_episodes) / len(selected_episodes)
                if selected_episodes else None
            ),
            "episode_false_safe_upper_95": _clopper_pearson_upper(
                len(failure_episodes), len(selected_episodes)
            ),
        }
    return {
        "proposal_count": len(rows),
        "strict_tau50_count": len(strict),
        "strict_tau50_rate": len(strict) / len(rows) if rows else 0.0,
        "strict_tau50_second_probe_restrictions": sum(labels),
        "scores": result,
    }


def _allocate_task_budget(rows: list[dict], trigger_count: int) -> dict[str, int]:
    """Allocate an integer fixed budget by task size without looking at scores."""

    task_counts = Counter(row["task"] for row in rows)
    if not task_counts or trigger_count <= 0:
        return {task: 0 for task in task_counts}
    trigger_count = min(trigger_count, len(rows))
    exact = {
        task: trigger_count * count / len(rows)
        for task, count in task_counts.items()
    }
    budget = {task: math.floor(value) for task, value in exact.items()}
    remaining = trigger_count - sum(budget.values())
    order = sorted(
        task_counts,
        key=lambda task: (exact[task] - budget[task], task),
        reverse=True,
    )
    for task in order[:remaining]:
        budget[task] += 1
    return budget


def compare_scores(rows: list[dict]) -> dict:
    labels = [bool(row["high_delayed_error"]) for row in rows]
    baseline_trigger_count = sum(
        float(row["global_mean"]) >= BASELINE_GATE_THRESHOLD for row in rows
    )
    target_trigger_count = max(1, round(len(rows) * TARGET_TRIGGER_RATE))
    task_budget = _allocate_task_budget(rows, target_trigger_count)
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
                rows, score, target_trigger_count
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
            train_trigger_count = max(1, round(len(train) * TARGET_TRIGGER_RATE))
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
            task_trigger_count = task_budget[task]
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
        "target_trigger_rate": TARGET_TRIGGER_RATE,
        "target_trigger_count": target_trigger_count,
        "realized_target_trigger_rate": target_trigger_count / len(rows),
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
    k2_rows: list[dict] = []
    audit = Counter()
    k2_audit = Counter()
    per_log = []
    for path in args.logs:
        log_rows, log_audit = audit_log(
            path, verify_threshold=args.verify_threshold
        )
        log_k2_rows, log_k2_audit = audit_second_probe_log(path)
        rows.extend(log_rows)
        k2_rows.extend(log_k2_rows)
        audit.update(log_audit)
        k2_audit.update(log_k2_audit)
        per_log.append(
            {
                "path": str(path),
                "audit": dict(log_audit),
                "second_probe_audit": dict(log_k2_audit),
            }
        )

    if not rows and not k2_rows:
        raise RuntimeError("no valid motion/error pairs or K2 proposals")

    identities = {
        (row["run"], row["task"], row["episode"], row["round_id"])
        for row in rows
    }
    if len(identities) != len(rows):
        raise RuntimeError("duplicate run/task/episode/round motion pairs")
    k2_identities = {
        (row["run"], row["task"], row["episode"], row["round_id"])
        for row in k2_rows
    }
    if len(k2_identities) != len(k2_rows):
        raise RuntimeError("duplicate run/task/episode/round K2 proposals")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_pairs(args.output_dir / "motion_pairs.csv", rows)
    write_pairs(args.output_dir / "motion_k2_proposals.csv", k2_rows)
    report = {
        "audit": dict(audit),
        "second_probe_audit": dict(k2_audit),
        "per_log": per_log,
        "comparison": compare_scores(rows) if rows else None,
        "second_probe_comparison": compare_second_probe_scores(k2_rows),
    }
    (args.output_dir / "motion_score_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
