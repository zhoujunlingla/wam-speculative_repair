import json

from scripts.analyze_motion_score import audit_log, compare_scores


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


def test_compare_scores_matches_baseline_trigger_budget():
    rows = [
        {"task": "a", "global_mean": 1.3, "median": 0.9, "high_delayed_error": True},
        {"task": "a", "global_mean": 0.2, "median": 0.1, "high_delayed_error": False},
        {"task": "b", "global_mean": 0.8, "median": 1.0, "high_delayed_error": True},
        {"task": "b", "global_mean": 0.1, "median": 0.2, "high_delayed_error": False},
    ]

    report = compare_scores(rows)

    assert report["baseline_trigger_count"] == 1
    assert report["overall"]["global_mean"]["matched_budget"]["trigger_count"] == 1
    assert report["overall"]["median"]["matched_budget"]["trigger_count"] == 1
    assert report["overall"]["global_mean"]["macro_task_average_precision"] == 1.0
