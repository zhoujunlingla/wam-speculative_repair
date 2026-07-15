import json

import pytest

from scripts.compare_paired_runs import load_trace, first_difference
from evaluation.robotwin.scene_manifest import load_manifest, write_manifest_atomic


def test_trace_comparison_accepts_identical_behavior_with_shadow_telemetry(tmp_path):
    off = tmp_path / "off.jsonl"
    shadow = tmp_path / "shadow.jsonl"
    base = {
        "source": "draft_flash",
        "executed_action_hash": "action",
        "accepted_prefix": 32,
        "fallback_reason": None,
        "cache_hash": "cache",
        "frame_st_id": 4,
    }
    off.write_text(json.dumps(base) + "\n")
    shadow.write_text(json.dumps({
        **base,
        "adaptive_k_certificate_kind": "pass",
        "adaptive_k_certificate_matches_full": True,
    }) + "\n")

    off_decisions, off_caches, _, off_certificates = load_trace(off)
    shadow_decisions, shadow_caches, conflicts, shadow_certificates = \
        load_trace(shadow)

    assert first_difference(off_decisions, shadow_decisions) is None
    assert first_difference(off_caches, shadow_caches) is None
    assert conflicts == 0
    assert off_certificates == 0
    assert shadow_certificates == 1


def test_trace_comparison_detects_action_and_certificate_conflict(tmp_path):
    off = tmp_path / "off.jsonl"
    shadow = tmp_path / "shadow.jsonl"
    off.write_text(json.dumps({
        "source": "draft_flash", "executed_action_hash": "a",
        "accepted_prefix": 32, "cache_hash": "c", "frame_st_id": 0,
    }) + "\n")
    shadow.write_text(json.dumps({
        "source": "draft_flash", "executed_action_hash": "b",
        "accepted_prefix": 32, "cache_hash": "c", "frame_st_id": 0,
        "adaptive_k_certificate_kind": "pass",
        "adaptive_k_certificate_matches_full": False,
    }) + "\n")

    off_decisions, _, _, _ = load_trace(off)
    shadow_decisions, _, conflicts, certificates = load_trace(shadow)

    assert first_difference(off_decisions, shadow_decisions) is not None
    assert conflicts == 1
    assert certificates == 1


def test_trace_comparison_rejects_empty_or_unhashed_evidence(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty audit trace"):
        load_trace(empty)

    unhashed = tmp_path / "unhashed.jsonl"
    unhashed.write_text(json.dumps({
        "source": "teacher_full", "accepted_prefix": 32,
    }) + "\n")
    with pytest.raises(ValueError, match="missing action hash"):
        load_trace(unhashed)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text(json.dumps({
        "source": "draft_flash", "executed_action_hash": "action",
        "accepted_prefix": 32, "cache_hash": "cache",
        "adaptive_k_certificate_kind": "pass",
        "adaptive_k_certificate_matches_full": None,
    }) + "\n")
    with pytest.raises(ValueError, match="malformed adaptive-K certificate"):
        load_trace(malformed)

    orphan = tmp_path / "orphan.jsonl"
    orphan.write_text(json.dumps({
        "source": "draft_flash", "executed_action_hash": "action",
        "accepted_prefix": 32, "cache_hash": "cache",
        "adaptive_k_certificate_kind": None,
        "adaptive_k_certificate_matches_full": False,
    }) + "\n")
    with pytest.raises(ValueError, match="orphan adaptive-K match"):
        load_trace(orphan)

    unsupported = tmp_path / "unsupported.jsonl"
    unsupported.write_text(json.dumps({
        "source": "draft_flash", "executed_action_hash": "action",
        "accepted_prefix": 32, "cache_hash": "cache",
        "adaptive_k_certificate_kind": "fail",
        "adaptive_k_certificate_matches_full": True,
    }) + "\n")
    with pytest.raises(ValueError, match="unsupported adaptive-K"):
        load_trace(unsupported)


def test_manifest_validation_fails_closed_on_task_or_count(tmp_path):
    path = tmp_path / "manifest.json"
    write_manifest_atomic(path, {
        "schema_version": 1,
        "task_name": "hanging_mug",
        "task_config": "demo_clean",
        "episodes": [{"episode_index": 0}],
    })

    with pytest.raises(ValueError, match="task_name"):
        load_manifest(
            path, task_name="open_microwave",
            task_config="demo_clean", test_num=1,
        )
    with pytest.raises(ValueError, match="enough episodes"):
        load_manifest(
            path, task_name="hanging_mug",
            task_config="demo_clean", test_num=2,
        )
