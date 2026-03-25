"""Tests for layered powermove gate decisions."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.powermove_gates import PowermoveGateThresholds, build_powermove_gate_report


def _diagnostics_report(
    *,
    max_overlap_frames: int = 23,
    source_track_ids: list[int] | None = None,
    identity_pass: bool = True,
    inverted_pct: float = 13.0,
) -> dict:
    return {
        "sequence": {"uid": "vid.4"},
        "segment": {"uid": "seg", "local_start_frame": 530, "local_end_frame_exclusive": 621},
        "segment_summary": {
            "max_raw_overlap_frames": max_overlap_frames,
            "frames_short_of_benchmark_gate": max(0, 45 - max_overlap_frames),
            "best_candidate_frames": max_overlap_frames,
        },
        "candidate_windows": [
            {
                "local_start_frame": 530,
                "local_end_frame_exclusive": 530 + max_overlap_frames,
                "source_track_ids": source_track_ids or [1],
                "josh": {"identity": {"pass": identity_pass}, "inversion": {"inverted_pct": inverted_pct}},
            }
        ]
        if max_overlap_frames > 0
        else [],
    }


def _root_cause_report(
    *,
    control_raw: float = 0.1063,
    target_raw: float = 2.2126,
    target_alt_px: float = 1199.64,
    target_px: float = 1199.64,
    target_center: float = 1.95,
    target_scale: float = 2.4142,
    target_offscreen: float = 0.8031,
    target_sim: float = 0.165,
    control_sim: float = 0.0826,
) -> dict:
    return {
        "target_window": {
            "diagnostics": {
                "josh_default": {
                    "mean_error_px": target_px,
                    "mean_error_bbox_diag_frac": target_raw,
                    "similarity_aligned_error_bbox_diag_frac": target_sim,
                    "mean_center_offset_bbox_diag_frac": target_center,
                    "mean_scale_ratio_pred_over_gt": target_scale,
                    "fraction_joints_out_of_frame": target_offscreen,
                },
                "josh_alt_cameraK": {
                    "mean_error_px": target_alt_px,
                },
            }
        },
        "control_window": {
            "diagnostics": {
                "josh_default": {
                    "mean_error_bbox_diag_frac": control_raw,
                    "similarity_aligned_error_bbox_diag_frac": control_sim,
                }
            }
        },
    }


def test_build_powermove_gate_report_classifies_systematic_placement_failure():
    report = build_powermove_gate_report(
        diagnostics_report=_diagnostics_report(),
        root_cause_report=_root_cause_report(),
    )

    assert report["gates"]["G1_application"]["status"] == "pass"
    assert report["gates"]["G2_extraction"]["status"] == "pass"
    assert report["gates"]["G3_placement"]["status"] == "fail"
    assert report["final_decision"]["classification"] == "systematic_model_placement_failure"
    assert report["final_decision"]["localized_rerun_allowed"] is False


def test_build_powermove_gate_report_flags_application_issue_when_control_is_bad():
    report = build_powermove_gate_report(
        diagnostics_report=_diagnostics_report(),
        root_cause_report=_root_cause_report(control_raw=0.35, target_alt_px=900.0),
    )

    assert report["gates"]["G1_application"]["status"] == "fail"
    assert report["final_decision"]["classification"] == "application_layer_issue"


def test_build_powermove_gate_report_allows_localized_rerun_only_after_placement_and_pose_pass():
    report = build_powermove_gate_report(
        diagnostics_report=_diagnostics_report(max_overlap_frames=30),
        root_cause_report=_root_cause_report(
            target_raw=0.18,
            target_center=0.2,
            target_scale=1.02,
            target_offscreen=0.05,
            target_sim=0.1,
            control_sim=0.08,
        ),
        thresholds=PowermoveGateThresholds(),
    )

    assert report["gates"]["G3_placement"]["status"] == "pass"
    assert report["gates"]["G4_pose"]["status"] == "pass"
    assert report["gates"]["G5_segment_viability"]["status"] == "fail"
    assert report["final_decision"]["classification"] == "coverage_only_after_local_rescue"
    assert report["final_decision"]["localized_rerun_allowed"] is True
    assert report["final_decision"]["broad_rerun_allowed"] is False
