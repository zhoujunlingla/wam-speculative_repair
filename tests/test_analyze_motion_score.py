import json

from scripts.analyze_motion_score import (
    average_precision,
    audit_log,
    audit_second_probe_log,
    compare_second_probe_scores,
    compare_scores,
)


def _write(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_audit_includes_only_executed_aligned_draft(tmp_path):
    log = tmp_path / "specverify_task.jsonl"
    motion = {
        "global_mean": 1.3,
        "median": 0.7,
        "top_mean": 2.0,
        "top_relative": 2.8,
        "top_concentration": 0.2,
    }
    _write(
        log,
        [
            {"source": "reset", "round_id": 0},
            {
                "source": "replan",
                "round_id": 1,
                "accepted_prefix": 0,
                "video_motion_stats": motion,
            },
            {
                "source": "draft_flash",
                "round_id": 2,
                "accepted_prefix": 16,
                "accepted_prefix_before_motion_cap": 32,
                "verify_threshold": 0.15,
                "prefix_by_tau": [32, 32],
                "verify_distances": [
                    [[0.01] * 16, [0.02] * 16],
                    [[0.03] * 16, [0.04] * 16],
                ],
                "video_motion_stats": motion,
            },
            {
                "source": "flash_cache_update",
                "round_id": 2,
                "cache_frame_count": 1,
                "delayed_video_error": {"latent_nrmse": 0.6},
            },
        ],
    )

    rows, audit = audit_log(log, run_name="run")

    assert len(rows) == 1
    assert rows[0]["episode"] == 0
    assert rows[0]["high_delayed_error"] is True
    assert rows[0]["accepted_prefix_before_motion_cap"] == 32
    assert rows[0]["verifier_tail_max"] == 0.04
    assert rows[0]["verifier_tail_strong"] is True
    assert audit["included_pair"] == 1
    assert audit["included_verifier_margin"] == 1
    assert audit["excluded_unexecuted_motion_proposal"] == 1


def test_audit_keeps_pair_when_verifier_margin_is_missing(tmp_path):
    log = tmp_path / "specverify_task.jsonl"
    motion = {"global_mean": 0.4, "median": 0.2}
    _write(
        log,
        [
            {"source": "reset", "round_id": 0},
            {
                "source": "draft_flash",
                "round_id": 1,
                "accepted_prefix": 16,
                "video_motion_stats": motion,
            },
            {
                "source": "flash_cache_update",
                "round_id": 1,
                "cache_frame_count": 1,
                "delayed_video_error": {"latent_nrmse": 0.2},
            },
        ],
    )

    rows, audit = audit_log(log, run_name="run")

    assert len(rows) == 1
    assert "verifier_tail_max" not in rows[0]
    assert audit["missing_verifier_margin"] == 1


def test_audit_history_uses_only_contiguous_executed_drafts(tmp_path):
    log = tmp_path / "specverify_task.jsonl"

    def motion(left, right, head):
        return {
            "global_mean": 0.4,
            "median": 0.2,
            "motion_v3_saliency_score": max(left, right, head),
            "motion_v2_regions": {
                "left_wrist": {"saliency_weighted": left},
                "right_wrist": {"saliency_weighted": right},
                "head": {"saliency_weighted": head},
            },
        }

    records = [{"source": "reset", "round_id": 0}]
    for round_id, stats in ((1, motion(1.0, 2.0, 3.0)), (2, motion(2.0, 2.0, 3.0))):
        records.extend(
            [
                {
                    "source": "draft_flash",
                    "round_id": round_id,
                    "accepted_prefix": 16,
                    "video_motion_stats": stats,
                },
                {
                    "source": "flash_cache_update",
                    "round_id": round_id,
                    "cache_frame_count": 1,
                    "delayed_video_error": {"latent_nrmse": 0.2},
                },
            ]
        )
    records.extend(
        [
            {"source": "replan", "round_id": 3},
            {
                "source": "draft_flash",
                "round_id": 4,
                "accepted_prefix": 16,
                "video_motion_stats": motion(3.0, 2.0, 3.0),
            },
            {
                "source": "flash_cache_update",
                "round_id": 4,
                "cache_frame_count": 1,
                "delayed_video_error": {"latent_nrmse": 0.2},
            },
        ]
    )
    _write(log, records)

    rows, audit = audit_log(log, run_name="run")

    assert "motion_v3_history_innovation" not in rows[0]
    assert abs(rows[1]["motion_v3_history_innovation"] - 1.0 / 3.0) < 1e-8
    assert "motion_v3_history_innovation" not in rows[2]
    assert audit["included_motion_v3_history"] == 1


def test_audit_rejects_empty_verifier_margin_without_dropping_pair(tmp_path):
    log = tmp_path / "specverify_task.jsonl"
    _write(
        log,
        [
            {"source": "reset", "round_id": 0},
            {
                "source": "draft_flash",
                "round_id": 1,
                "accepted_prefix": 16,
                "verify_threshold": 0.15,
                "verify_distances": [],
                "video_motion_stats": {"global_mean": 0.4, "median": 0.2},
            },
            {
                "source": "flash_cache_update",
                "round_id": 1,
                "cache_frame_count": 1,
                "delayed_video_error": {"latent_nrmse": 0.2},
            },
        ],
    )

    rows, audit = audit_log(log, run_name="run")

    assert len(rows) == 1
    assert "verifier_tail_max" not in rows[0]
    assert audit["malformed_verifier_margin"] == 1


def test_audit_uses_explicit_threshold_for_legacy_log(tmp_path):
    log = tmp_path / "specverify_task.jsonl"
    _write(
        log,
        [
            {"source": "reset", "round_id": 0},
            {
                "source": "draft_flash",
                "round_id": 1,
                "accepted_prefix": 16,
                "verify_distances": [
                    [[0.01] * 16, [0.02] * 16],
                    [[0.03] * 16, [0.04] * 16],
                ],
                "video_motion_stats": {"global_mean": 0.4, "median": 0.2},
            },
            {
                "source": "flash_cache_update",
                "round_id": 1,
                "cache_frame_count": 1,
                "delayed_video_error": {"latent_nrmse": 0.2},
            },
        ],
    )

    rows, audit = audit_log(log, run_name="run", verify_threshold=0.15)

    assert rows[0]["verify_threshold"] == 0.15
    assert rows[0]["verifier_tail_margin"] == 1.0 - 0.04 / 0.15
    assert audit["included_verifier_margin"] == 1


def test_audit_rejects_frame_alignment_mismatch(tmp_path):
    log = tmp_path / "specverify_task.jsonl"
    motion = {
        "global_mean": 0.4,
        "median": 0.2,
        "top_mean": 1.0,
        "top_relative": 5.0,
        "top_concentration": 0.3,
    }
    _write(
        log,
        [
            {"source": "reset", "round_id": 0},
            {
                "source": "draft_flash",
                "round_id": 1,
                "accepted_prefix": 32,
                "video_motion_stats": motion,
            },
            {
                "source": "flash_cache_update",
                "round_id": 1,
                "cache_frame_count": 1,
                "delayed_video_error": {"latent_nrmse": 0.2},
            },
        ],
    )

    rows, audit = audit_log(log, run_name="run")

    assert rows == []
    assert audit["excluded_frame_alignment_mismatch"] == 1


def test_compare_scores_uses_fixed_trigger_budget():
    rows = [
        {"task": "a", "global_mean": 1.3, "median": 0.9, "high_delayed_error": True},
        {"task": "a", "global_mean": 0.2, "median": 0.1, "high_delayed_error": False},
        {"task": "b", "global_mean": 0.8, "median": 1.0, "high_delayed_error": True},
        {"task": "b", "global_mean": 0.1, "median": 0.2, "high_delayed_error": False},
    ]

    report = compare_scores(rows)

    assert report["target_trigger_count"] == 1
    assert report["overall"]["global_mean"]["matched_budget"]["trigger_count"] == 1
    assert report["overall"]["median"]["matched_budget"]["trigger_count"] == 1
    assert report["overall"]["global_mean"]["macro_task_average_precision"] == 1.0


def test_average_precision_is_invariant_to_tie_order():
    assert average_precision([1.0, 1.0, 0.0], [True, False, False]) == 0.5
    assert average_precision([1.0, 1.0, 0.0], [False, True, False]) == 0.5


def test_compare_scores_includes_complete_v3_saliency():
    rows = [
        {
            "task": "a",
            "global_mean": 1.0,
            "median": 0.5,
            "motion_v2_score": 0.6,
            "motion_v3_saliency_score": 0.7,
            "high_delayed_error": True,
        },
        {
            "task": "b",
            "global_mean": 0.5,
            "median": 0.3,
            "motion_v2_score": 0.4,
            "motion_v3_saliency_score": 0.2,
            "high_delayed_error": False,
        },
    ]

    report = compare_scores(rows)

    assert "motion_v3_saliency" in report["overall"]


def test_leave_one_task_out_threshold_does_not_use_heldout_scores():
    def rows(heldout_scores):
        result = []
        for task, scores in {
            "a": [0.9, 0.8],
            "b": [0.7, 0.6],
            "c": heldout_scores,
        }.items():
            for index, score in enumerate(scores):
                result.append(
                    {
                        "task": task,
                        "global_mean": score,
                        "median": score,
                        "high_delayed_error": index == 0,
                    }
                )
        return result

    first = compare_scores(rows([0.5, 0.4]))
    second = compare_scores(rows([100.0, -100.0]))
    first_fold = next(
        fold
        for fold in first["leave_one_task_out_threshold"]["global_mean"]["folds"]
        if fold["task"] == "c"
    )
    second_fold = next(
        fold
        for fold in second["leave_one_task_out_threshold"]["global_mean"]["folds"]
        if fold["task"] == "c"
    )

    assert first_fold["threshold"] == second_fold["threshold"]


def test_task_balanced_budget_sums_to_frozen_total():
    rows = [
        {
            "task": f"task_{index % 10}",
            "global_mean": float(index),
            "median": float(index),
            "high_delayed_error": index % 3 == 0,
        }
        for index in range(100)
    ]

    report = compare_scores(rows)
    budget = report["task_balanced_matched_budget"]["global_mean"]

    assert report["target_trigger_count"] == 7
    assert budget["trigger_count"] == 7


def test_second_probe_audit_includes_rejected_proposals(tmp_path):
    log = tmp_path / "specverify_task.jsonl"
    _write(
        log,
        [
            {"source": "reset", "round_id": 0},
            {
                "source": "replan",
                "round_id": 1,
                "accepted_prefix": 0,
                "active_tau_timesteps": [50, 100],
                "prefix_by_tau": [32, 28],
                "gripper_phase_agreement_by_tau": [1.0, 1.0],
                "draft_gripper_switch_index": None,
                "gripper_switch_indices_by_tau": [None, None],
                "verify_distances": [
                    [[0.01] * 16, [0.02] * 16],
                    [[0.03] * 16, [0.04] * 16],
                ],
                "video_motion_stats": {"global_mean": 0.4, "median": 0.2},
            },
        ],
    )

    rows, audit = audit_second_probe_log(log, run_name="run")

    assert len(rows) == 1
    assert rows[0]["tau50_prefix"] == 32
    assert rows[0]["tau100_prefix"] == 16
    assert rows[0]["second_probe_restricts_prefix"] is True
    assert rows[0]["second_probe_restricts"] is True
    assert audit["included_k2_proposal"] == 1


def test_second_probe_audit_detects_phase_restriction(tmp_path):
    log = tmp_path / "specverify_task.jsonl"
    _write(
        log,
        [
            {
                "source": "draft_flash",
                "round_id": 1,
                "active_tau_timesteps": [50, 100],
                "prefix_by_tau": [32, 32],
                "gripper_phase_agreement_by_tau": [1.0, 0.75],
                "draft_gripper_switch_index": None,
                "gripper_switch_indices_by_tau": [None, 8],
                "verify_distances": [
                    [[0.01] * 16, [0.02] * 16],
                    [[0.03] * 16, [0.04] * 16],
                ],
                "video_motion_stats": {"global_mean": 0.4, "median": 0.2},
            }
        ],
    )

    rows, _ = audit_second_probe_log(log)

    assert rows[0]["second_probe_restricts_prefix"] is False
    assert rows[0]["second_probe_restricts_phase"] is True
    assert rows[0]["second_probe_restricts"] is True


def test_second_probe_audit_excludes_missing_and_nonstandard_telemetry(tmp_path):
    log = tmp_path / "specverify_task.jsonl"
    base = {
        "source": "draft_flash",
        "prefix_by_tau": [32, 32],
        "gripper_phase_agreement_by_tau": [1.0, 1.0],
        "draft_gripper_switch_index": None,
        "gripper_switch_indices_by_tau": [None, None],
        "verify_distances": [
            [[0.01] * 16, [0.02] * 16],
            [[0.03] * 16, [0.04] * 16],
        ],
        "video_motion_stats": {"global_mean": 0.4, "median": 0.2},
    }
    _write(
        log,
        [
            {**base, "round_id": 1},
            {**base, "round_id": 2, "active_tau_timesteps": [150, 300]},
        ],
    )

    rows, audit = audit_second_probe_log(log)

    assert rows == []
    assert audit["excluded_missing_k2_telemetry"] == 1
    assert audit["excluded_nonstandard_k2_probe"] == 1


def test_second_probe_comparison_uses_task_heldout_zero_miss_threshold():
    def row(task, motion, restricts):
        return {
            "task": task,
            "tau50_prefix": 32,
            "tau50_max_residual": 0.01,
            "tau50_gripper_stable": True,
            "tau50_no_phase_switch": True,
            "second_probe_restricts": restricts,
            "run": "run",
            "episode": 0 if task == "a" else 1,
            "global_mean": motion,
            "median": motion,
        }

    rows = [
        row("a", 0.1, False),
        row("a", 0.8, True),
        row("b", 0.2, False),
        row("b", 0.8, True),
    ]

    report = compare_second_probe_scores(rows)
    score = report["scores"]["global_mean"]

    assert report["proposal_count"] == 4
    assert report["strict_tau50_count"] == 4
    assert score["selected_count"] == 2
    assert score["failures"] == 0
    assert score["coverage"] == 0.5
    assert score["false_safe_upper_95"] is not None


def test_second_probe_comparison_abstains_without_training_positive():
    rows = [
        {
            "task": task,
            "run": "run",
            "episode": index,
            "tau50_prefix": 32,
            "tau50_max_residual": 0.01,
            "tau50_gripper_stable": True,
            "tau50_no_phase_switch": True,
            "second_probe_restricts": False,
            "global_mean": 0.1,
            "median": 0.1,
        }
        for index, task in enumerate(("a", "b"))
    ]

    report = compare_second_probe_scores(rows)

    assert report["scores"]["global_mean"]["selected_count"] == 0
    assert all(
        fold["insufficient_positive_support"]
        for fold in report["scores"]["global_mean"]["folds"]
    )
