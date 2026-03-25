"""Layered no-rerun gates for JOSH powermove diagnosis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PowermoveGateThresholds:
    intrinsics_rel_tol: float = 0.01
    control_max_raw_bbox_diag_frac: float = 0.15
    placement_max_raw_bbox_diag_frac: float = 0.25
    placement_max_center_offset_bbox_diag_frac: float = 0.35
    placement_scale_ratio_min: float = 0.8
    placement_scale_ratio_max: float = 1.25
    placement_max_offscreen_fraction: float = 0.15
    pose_max_similarity_bbox_diag_frac: float = 0.12
    pose_max_delta_vs_control_bbox_diag_frac: float = 0.03
    benchmark_min_frames: int = 45


def _safe_rel_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if abs(a) < 1e-8:
        return 0.0 if abs(b) < 1e-8 else None
    return abs(a - b) / abs(a)


def _gate(status: str, *, verdict: str, reason: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": status,
        "verdict": verdict,
        "reason": reason,
        "evidence": evidence,
    }


def build_powermove_gate_report(
    *,
    diagnostics_report: dict[str, Any],
    root_cause_report: dict[str, Any],
    thresholds: PowermoveGateThresholds | None = None,
) -> dict[str, Any]:
    """Combine focused diagnostics and root-cause metrics into one gate verdict."""
    thresholds = thresholds or PowermoveGateThresholds()
    candidate = diagnostics_report["candidate_windows"][0] if diagnostics_report["candidate_windows"] else None
    target = root_cause_report["target_window"]["diagnostics"]["josh_default"]
    control = root_cause_report["control_window"]["diagnostics"]["josh_default"]
    altk = root_cause_report["target_window"]["diagnostics"]["josh_alt_cameraK"]
    segment_summary = diagnostics_report["segment_summary"]

    altk_rel_delta = _safe_rel_delta(target["mean_error_px"], altk["mean_error_px"])
    gate_application = _gate(
        "pass"
        if control["mean_error_bbox_diag_frac"] is not None
        and control["mean_error_bbox_diag_frac"] <= thresholds.control_max_raw_bbox_diag_frac
        and altk_rel_delta is not None
        and altk_rel_delta <= thresholds.intrinsics_rel_tol
        else "fail",
        verdict="application_layer_falsified"
        if control["mean_error_bbox_diag_frac"] is not None
        and control["mean_error_bbox_diag_frac"] <= thresholds.control_max_raw_bbox_diag_frac
        and altk_rel_delta is not None
        and altk_rel_delta <= thresholds.intrinsics_rel_tol
        else "application_layer_still_live",
        reason=(
            "The same projection/evaluation path works on the footwork control and changing intrinsics does not materially change the powermove error."
            if control["mean_error_bbox_diag_frac"] is not None
            and control["mean_error_bbox_diag_frac"] <= thresholds.control_max_raw_bbox_diag_frac
            and altk_rel_delta is not None
            and altk_rel_delta <= thresholds.intrinsics_rel_tol
            else "Either the footwork control is not healthy or alternate intrinsics still move the powermove result materially."
        ),
        evidence={
            "control_raw_bbox_diag_frac": control["mean_error_bbox_diag_frac"],
            "target_raw_error_px": target["mean_error_px"],
            "target_alt_cameraK_error_px": altk["mean_error_px"],
            "target_alt_cameraK_rel_delta": None if altk_rel_delta is None else round(float(altk_rel_delta), 6),
        },
    )

    extraction_ok = bool(
        candidate is not None
        and candidate["source_track_ids"]
        and len(candidate["source_track_ids"]) == 1
        and candidate["josh"]["identity"]["pass"]
    )
    gate_extraction = _gate(
        "pass" if extraction_ok else "fail",
        verdict="extraction_not_primary_on_surviving_slice" if extraction_ok else "extraction_or_assembly_issue",
        reason=(
            "The surviving slice is contiguous, single-track, and identity-clean, so extraction is not the lead suspect on that slice."
            if extraction_ok
            else "No clean surviving candidate exists or the candidate still shows track/identity problems."
        ),
        evidence={
            "candidate_window_count": len(diagnostics_report["candidate_windows"]),
            "best_candidate_frames": segment_summary["best_candidate_frames"],
            "source_track_ids": candidate["source_track_ids"] if candidate is not None else [],
            "identity_pass": candidate["josh"]["identity"]["pass"] if candidate is not None else False,
        },
    )

    placement_ok = bool(
        target["mean_error_bbox_diag_frac"] is not None
        and target["mean_error_bbox_diag_frac"] <= thresholds.placement_max_raw_bbox_diag_frac
        and target["mean_center_offset_bbox_diag_frac"] is not None
        and target["mean_center_offset_bbox_diag_frac"] <= thresholds.placement_max_center_offset_bbox_diag_frac
        and target["mean_scale_ratio_pred_over_gt"] is not None
        and thresholds.placement_scale_ratio_min
        <= target["mean_scale_ratio_pred_over_gt"]
        <= thresholds.placement_scale_ratio_max
        and target["fraction_joints_out_of_frame"] is not None
        and target["fraction_joints_out_of_frame"] <= thresholds.placement_max_offscreen_fraction
    )
    gate_placement = _gate(
        "pass" if placement_ok else "fail",
        verdict="placement_scale_rescued" if placement_ok else "systematic_model_placement_failure",
        reason=(
            "The target slice now lands in the same geometric regime as the control."
            if placement_ok
            else "Raw error is still dominated by bad camera-relative placement/scale on the target slice."
        ),
        evidence={
            "target_raw_bbox_diag_frac": target["mean_error_bbox_diag_frac"],
            "target_center_offset_bbox_diag_frac": target["mean_center_offset_bbox_diag_frac"],
            "target_scale_ratio_pred_over_gt": target["mean_scale_ratio_pred_over_gt"],
            "target_fraction_joints_out_of_frame": target["fraction_joints_out_of_frame"],
        },
    )

    pose_ok = bool(
        target["similarity_aligned_error_bbox_diag_frac"] is not None
        and control["similarity_aligned_error_bbox_diag_frac"] is not None
        and target["similarity_aligned_error_bbox_diag_frac"] <= thresholds.pose_max_similarity_bbox_diag_frac
        and target["similarity_aligned_error_bbox_diag_frac"]
        <= control["similarity_aligned_error_bbox_diag_frac"] + thresholds.pose_max_delta_vs_control_bbox_diag_frac
    )
    gate_pose = _gate(
        "pass" if pose_ok else "fail",
        verdict="residual_pose_ok" if pose_ok else "model_pose_failure_after_placement",
        reason=(
            "After similarity alignment, the target slice is now close enough to the control regime."
            if pose_ok
            else "Even after removing placement/scale, the target slice still shows materially worse residual pose error than the control."
        ),
        evidence={
            "target_similarity_bbox_diag_frac": target["similarity_aligned_error_bbox_diag_frac"],
            "control_similarity_bbox_diag_frac": control["similarity_aligned_error_bbox_diag_frac"],
            "target_inverted_pct": candidate["josh"]["inversion"]["inverted_pct"] if candidate is not None else None,
        },
    )

    viability_ok = segment_summary["max_raw_overlap_frames"] >= thresholds.benchmark_min_frames
    gate_viability = _gate(
        "pass" if viability_ok else "fail",
        verdict="benchmarkable_window_available" if viability_ok else "still_not_segment_viable",
        reason=(
            "The segment now has a contiguous JOSH run long enough for the benchmark gate."
            if viability_ok
            else "The segment still lacks a contiguous run that reaches the benchmark gate."
        ),
        evidence={
            "max_raw_overlap_frames": segment_summary["max_raw_overlap_frames"],
            "benchmark_min_frames": thresholds.benchmark_min_frames,
            "frames_short_of_benchmark_gate": segment_summary["frames_short_of_benchmark_gate"],
        },
    )

    gate_order = [
        ("G1_application", gate_application),
        ("G2_extraction", gate_extraction),
        ("G3_placement", gate_placement),
        ("G4_pose", gate_pose),
        ("G5_segment_viability", gate_viability),
    ]

    if gate_application["status"] == "fail":
        final_classification = "application_layer_issue"
        next_step = "Fix projection/evaluation invariants before touching JOSH behavior."
    elif gate_extraction["status"] == "fail":
        final_classification = "extraction_or_assembly_issue"
        next_step = "Stabilize the surviving slice first; do not interpret model quality until a clean candidate exists."
    elif gate_placement["status"] == "fail":
        final_classification = "systematic_model_placement_failure"
        next_step = "Target JOSH camera-relative placement/scale on the control slice before any rerun."
    elif gate_pose["status"] == "fail":
        final_classification = "model_pose_failure_after_placement"
        next_step = "Test a stronger prior on the same control slice before any rerun."
    elif gate_viability["status"] == "fail":
        final_classification = "coverage_only_after_local_rescue"
        next_step = "Only now consider a localized rerun or continuity-focused tuning."
    else:
        final_classification = "segment_ready"
        next_step = "The control slice is healthy enough to justify broader validation."

    localized_rerun_allowed = (
        gate_application["status"] == "pass"
        and gate_placement["status"] == "pass"
        and gate_pose["status"] == "pass"
    )
    broad_rerun_allowed = localized_rerun_allowed and gate_viability["status"] == "pass"

    return {
        "sequence": diagnostics_report["sequence"],
        "segment": diagnostics_report["segment"],
        "control_window": root_cause_report["control_window"],
        "thresholds": asdict(thresholds),
        "gates": {name: payload for name, payload in gate_order},
        "final_decision": {
            "classification": final_classification,
            "localized_rerun_allowed": localized_rerun_allowed,
            "broad_rerun_allowed": broad_rerun_allowed,
            "next_step": next_step,
        },
        "evidence_snapshot": {
            "segment_summary": diagnostics_report["segment_summary"],
            "target_root_cause": target,
            "control_root_cause": control,
        },
    }


def render_powermove_gate_markdown(report: dict[str, Any]) -> str:
    """Render a readable markdown summary for the layered gates."""
    segment = report["segment"]
    decision = report["final_decision"]
    lines = [
        "# JOSH Powermove Layered Gates",
        "",
        "## Summary",
        "",
        f"- Segment: `{segment['uid']}`",
        f"- Classification: `{decision['classification']}`",
        f"- Localized rerun allowed: `{decision['localized_rerun_allowed']}`",
        f"- Broad rerun allowed: `{decision['broad_rerun_allowed']}`",
        f"- Next step: {decision['next_step']}",
        "",
        "## Gates",
        "",
        "| Gate | Status | Verdict | Reason |",
        "|------|--------|---------|--------|",
    ]
    for gate_name, payload in report["gates"].items():
        lines.append(
            f"| `{gate_name}` | `{payload['status']}` | `{payload['verdict']}` | {payload['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Key Evidence",
            "",
            f"- Target raw bbox-diag error: `{report['evidence_snapshot']['target_root_cause']['mean_error_bbox_diag_frac']}`",
            f"- Target similarity-aligned bbox-diag error: `{report['evidence_snapshot']['target_root_cause']['similarity_aligned_error_bbox_diag_frac']}`",
            f"- Target center offset bbox-diag: `{report['evidence_snapshot']['target_root_cause']['mean_center_offset_bbox_diag_frac']}`",
            f"- Target scale ratio: `{report['evidence_snapshot']['target_root_cause']['mean_scale_ratio_pred_over_gt']}`",
            f"- Target out-of-frame fraction: `{report['evidence_snapshot']['target_root_cause']['fraction_joints_out_of_frame']}`",
            f"- Max contiguous JOSH overlap: `{report['evidence_snapshot']['segment_summary']['max_raw_overlap_frames']}` frames",
        ]
    )
    return "\n".join(lines) + "\n"


def write_powermove_gate_outputs(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write the machine-readable and markdown gate reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "gates_report.json"
    md_path = output_dir / "gates_report.md"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(md_path, "w") as f:
        f.write(render_powermove_gate_markdown(report))
    return {"json": str(json_path), "markdown": str(md_path)}
