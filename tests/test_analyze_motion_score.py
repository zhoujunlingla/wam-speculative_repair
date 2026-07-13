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
    assert audit["included_pair"] == 1
    assert audit["excluded_unexecuted_motion_proposal"] == 1


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
