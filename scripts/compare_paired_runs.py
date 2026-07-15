#!/usr/bin/env python3
"""Fail closed unless adaptive-K off and shadow traces are behavior-identical."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DECISION_SOURCES = {"teacher_full", "draft_flash", "replan"}


def load_trace(path: Path) -> tuple[list[dict], list[dict], int, int]:
    decisions = []
    caches = []
    conflicts = 0
    certificates = 0
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"empty audit trace: {path}")
    for line_number, line in enumerate(lines, 1):
        row = json.loads(line)
        certificate_kind = row.get("adaptive_k_certificate_kind")
        certificate_match = row.get("adaptive_k_certificate_matches_full")
        if certificate_kind is None:
            if certificate_match is not None:
                raise ValueError(
                    f"orphan adaptive-K match in {path}:{line_number}"
                )
        elif certificate_kind == "pass":
            if type(certificate_match) is not bool:
                raise ValueError(
                    f"malformed adaptive-K certificate in {path}:{line_number}"
                )
            certificates += 1
            conflicts += int(not certificate_match)
        else:
            raise ValueError(
                f"unsupported adaptive-K certificate kind in {path}:{line_number}"
            )
        if row.get("source") in DECISION_SOURCES:
            if (
                row.get("source") != "replan"
                and not row.get("executed_action_hash")
            ):
                raise ValueError(
                    f"missing action hash in {path}:{line_number}"
                )
            decisions.append({
                "source": row.get("source"),
                "executed_action_hash": row.get("executed_action_hash"),
                "accepted_prefix": row.get("accepted_prefix"),
                "fallback_reason": row.get("fallback_reason"),
            })
        if row.get("cache_hash") is None:
            raise ValueError(f"missing cache hash in {path}:{line_number}")
        caches.append({
            "source": row.get("source"),
            "cache_hash": row["cache_hash"],
            "frame_st_id": row.get("frame_st_id"),
        })
    if not decisions:
        raise ValueError(f"audit trace has no action decisions: {path}")
    return decisions, caches, conflicts, certificates


def first_difference(left: list[dict], right: list[dict]):
    for index, (left_row, right_row) in enumerate(zip(left, right)):
        if left_row != right_row:
            return {"index": index, "off": left_row, "shadow": right_row}
    if len(left) != len(right):
        return {"index": min(len(left), len(right)),
                "off_length": len(left), "shadow_length": len(right)}
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    off_decisions, off_caches, off_conflicts, off_certificates = \
        load_trace(args.off)
    shadow_decisions, shadow_caches, shadow_conflicts, shadow_certificates = \
        load_trace(args.shadow)
    report = {
        "off": str(args.off),
        "shadow": str(args.shadow),
        "decision_rounds": len(off_decisions),
        "cache_records": len(off_caches),
        "decision_difference": first_difference(off_decisions, shadow_decisions),
        "cache_difference": first_difference(off_caches, shadow_caches),
        "off_certificate_conflicts": off_conflicts,
        "shadow_certificate_conflicts": shadow_conflicts,
        "off_certificates": off_certificates,
        "shadow_certificates": shadow_certificates,
    }
    report["equivalent"] = bool(
        report["decision_difference"] is None
        and report["cache_difference"] is None
        and off_conflicts == 0
        and off_certificates == 0
        and shadow_conflicts == 0
        and shadow_certificates > 0
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    if not report["equivalent"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
