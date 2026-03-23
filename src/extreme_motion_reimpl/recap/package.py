"""Bundle recap outputs into a shareable package."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path


def generate_summary(metrics: dict, audio: dict, output_path: Path) -> None:
    """Generate human-readable summary.txt."""
    m = metrics["musicality"]
    e = metrics["energy"]
    f = metrics["flow"]
    s = metrics["space"]
    c = metrics["complexity"]
    meta = metrics["meta"]

    lines = [
        "=" * 50,
        "  BATTLE ANALYTICS RECAP",
        "=" * 50,
        "",
        f"  Duration: {meta['duration_s']}s  |  {meta['n_frames']} frames @ {meta['fps']} FPS",
        f"  Joints:   {meta['n_joints']}",
        "",
        "── Musicality ─────────────────────────────────",
        f"  Score (μ):        {m['mu']:.3f}  [{m['interpretation']['musicality']}]",
        f"  Timing (τ*):      {m['tau_star_ms']:.0f} ms  [{m['interpretation']['timing']}]",
        f"  Beat alignment:   {m['beat_alignment_pct']:.0f}%",
        "",
        "── Energy ─────────────────────────────────────",
        f"  Mean energy:      {e['mean_energy']:.4f}",
        f"  Peak energy:      {e['max_energy']:.4f}",
        f"  Build-up rate:    {e['build_up_rate']:.4f}",
        f"  Peak moments:     {len(e['peak_moments'])}",
        "",
        "── Flow ───────────────────────────────────────",
        f"  Flow score:       {f['flow_score']:.1f} / 100",
        f"  Mean jerk:        {f['mean_jerk']:.4f}",
        "",
        "── Space ──────────────────────────────────────",
        f"  Stage coverage:   {s['stage_coverage_m2']:.2f} m²",
        f"  Vertical range:   {s['vertical_range_m']:.2f} m",
        f"  Travel speed:     {s['mean_travel_speed_m_s']:.3f} m/s",
        "",
        "── Complexity ─────────────────────────────────",
        f"  Freezes:          {c['freeze_count']} ({c['total_freeze_time_s']:.1f}s total)",
        f"  Inversions:       {c['inversion_count']}",
        f"  Peak accel:       {c['peak_acceleration_m_s2']:.1f} m/s²",
        "",
        "── Audio ──────────────────────────────────────",
        f"  BPM:              {audio.get('bpm', '?')}",
        f"  BPM stability:    {audio.get('bpm_stability', '?')}",
        f"  Beats detected:   {audio.get('n_beats', '?')}",
        f"  Source:           {audio.get('source', '?')}",
        "",
        "=" * 50,
    ]

    output_path.write_text("\n".join(lines))


def create_zip(output_dir: Path, zip_path: Path | None = None) -> Path:
    """Zip the recap directory."""
    if zip_path is None:
        zip_path = output_dir.parent / f"{output_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(output_dir.rglob("*")):
            if f.is_file() and f.suffix != ".zip":
                zf.write(f, f.relative_to(output_dir))
    return zip_path
