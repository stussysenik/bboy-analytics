"""
Experiment harness: run a configured experiment, save all outputs,
and return structured results for journal entry.

Key design: joint generation beats and evaluation beats are DECOUPLED.
Controls use the same joints (generated with real BRACE beats) but
evaluate against shifted or random beat sets.
"""

from __future__ import annotations

import json
import time
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from extreme_motion_reimpl.recap.metrics import compute_all_metrics, compute_musicality
from extreme_motion_reimpl.recap.render import render_all
from experiments.synthetic_joints import (
    generate_toprock_onbeat,
    generate_random_control,
    generate_powermove,
    load_brace_beats,
    load_brace_segments,
)


def run_experiment(config: dict) -> dict:
    """
    Run a single experiment from config dict.

    Config fields:
        experiment_id: str
        title: str
        joint_source: "toprock_onbeat" | "random_control" | "powermove" | "synthetic_default"
        video_id: str (BRACE video ID)
        seq_idx: int (BRACE sequence index)
        fps: float (default 30.0)
        sg_window: int (default 31)
        seed: int (default 42)
        duration_s: float (optional)
        # Beat control:
        eval_beat_source: "brace" | "shifted" | "random" | "synthetic_120bpm"
        eval_beat_shift_s: float (for "shifted" source)
        # Joint generation always uses BRACE beats when video_id is set
    """
    exp_id = config["experiment_id"]
    title = config["title"]
    fps = config.get("fps", 30.0)
    sg_window = config.get("sg_window", 31)
    seed = config.get("seed", 42)

    slug = f"{exp_id}_{title.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')[:40]}"
    out_dir = Path(__file__).parent / "results" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {exp_id}: {title}")
    print(f"{'='*60}")

    t0 = time.time()
    video_id = config.get("video_id")
    seq_idx = config.get("seq_idx", 0)

    # ── Load BRACE beats (for joint generation) ─────────────────
    brace_data = None
    gen_beats = None
    if video_id:
        brace_data = load_brace_beats(video_id, seq_idx)
        gen_beats = brace_data["beat_times"]
        print(f"  BRACE: {video_id}.{seq_idx} — {brace_data['bpm']:.1f} BPM, {len(gen_beats)} beats, conf={brace_data['confidence']:.1f}")

    # ── Determine EVALUATION beats (may differ from gen beats) ──
    eval_beat_source = config.get("eval_beat_source", "brace")
    eval_beat_shift = config.get("eval_beat_shift_s", 0.0)

    if eval_beat_source == "brace" and gen_beats is not None:
        eval_beats = gen_beats.copy()
        print(f"  Eval beats: BRACE (same as generation)")
    elif eval_beat_source == "shifted" and gen_beats is not None:
        eval_beats = gen_beats + eval_beat_shift
        print(f"  Eval beats: SHIFTED by {eval_beat_shift*1000:.0f} ms")
    elif eval_beat_source == "random":
        rng_beats = np.random.default_rng(seed + 999)
        duration = gen_beats[-1] + 2.0 if gen_beats is not None else 30.0
        n_eval = len(gen_beats) if gen_beats is not None else 60
        eval_beats = np.sort(rng_beats.uniform(0, duration, n_eval))
        print(f"  Eval beats: RANDOM — {n_eval} uniformly distributed")
    else:
        duration = config.get("duration_s", 30.0)
        eval_beats = np.arange(0, duration, 0.5)
        print(f"  Eval beats: SYNTHETIC 120 BPM")

    # ── Generate joints ─────────────────────────────────────────
    joint_source = config.get("joint_source", "toprock_onbeat")
    duration_s = config.get("duration_s")

    if joint_source == "toprock_onbeat":
        # Always generate with BRACE beats (or eval beats if no BRACE)
        jbeats = gen_beats if gen_beats is not None else eval_beats
        joints = generate_toprock_onbeat(jbeats, fps, duration_s, seed=seed)
        print(f"  Joints: toprock on-beat, {joints.shape[0]} frames ({joints.shape[0]/fps:.1f}s)")
    elif joint_source == "random_control":
        dur = duration_s or (gen_beats[-1] + 2.0 if gen_beats is not None else 30.0)
        joints = generate_random_control(fps, dur, seed=seed)
        print(f"  Joints: random control (no beat structure), {joints.shape[0]} frames")
    elif joint_source == "powermove":
        jbeats = gen_beats if gen_beats is not None else eval_beats
        joints = generate_powermove(jbeats, fps, duration_s, seed=seed)
        print(f"  Joints: powermove, {joints.shape[0]} frames")
    elif joint_source == "synthetic_default":
        n = int((duration_s or 30.0) * fps)
        rng = np.random.default_rng(seed)
        joints = np.tile(np.array([SMPL_DEFAULT_POSE], dtype=np.float64), (n, 1, 1))
        t = np.arange(n) / fps
        for j in range(22):
            joints[:, j, 0] += 0.03 * np.sin(2 * np.pi * 2 * t + rng.uniform(0, np.pi))
            joints[:, j, 1] += 0.02 * np.sin(2 * np.pi * 2 * t + rng.uniform(0, np.pi))
        print(f"  Joints: synthetic default, {joints.shape[0]} frames")
    else:
        raise ValueError(f"Unknown joint_source: {joint_source}")

    np.save(out_dir / "joints_3d.npy", joints)

    # ── Load segments ───────────────────────────────────────────
    segments = None
    if video_id:
        try:
            segments = load_brace_segments(video_id, seq_idx)
            if segments:
                print(f"  Segments: {len(segments)} ({', '.join(s['dance_type'] for s in segments)})")
        except Exception:
            pass

    # ── Compute metrics with EVALUATION beats ───────────────────
    metrics = compute_all_metrics(joints, fps, eval_beats, segments)

    if sg_window != 31:
        metrics["musicality"] = compute_musicality(joints, eval_beats, fps, sg_window=sg_window)

    elapsed = time.time() - t0

    # ── Save outputs ────────────────────────────────────────────
    metrics_clean = _clean_for_json(metrics)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_clean, f, indent=2)

    try:
        created = render_all(metrics, out_dir, fps)
        print(f"  Rendered: {', '.join(created)}")
    except Exception as e:
        print(f"  WARNING: Render failed: {e}")
        created = []

    meta = {
        "experiment_id": exp_id,
        "title": title,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "wall_time_s": round(elapsed, 2),
        "joint_source": joint_source,
        "eval_beat_source": eval_beat_source,
        "eval_beat_shift_s": eval_beat_shift,
        "video_id": video_id,
        "seq_idx": seq_idx,
        "fps": fps,
        "sg_window": sg_window,
        "seed": seed,
        "n_frames": joints.shape[0],
        "duration_s": round(joints.shape[0] / fps, 2),
        "brace_bpm": brace_data["bpm"] if brace_data else None,
        "brace_confidence": brace_data["confidence"] if brace_data else None,
        "n_eval_beats": len(eval_beats),
        "rendered_files": created,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── Print summary ───────────────────────────────────────────
    mu = metrics["musicality"]["mu"]
    tau = metrics["musicality"]["tau_star_ms"]
    interp = metrics["musicality"]["interpretation"]["musicality"]
    flow = metrics["flow"]["flow_score"]
    coverage = metrics["space"]["stage_coverage_m2"]
    beat_pct = metrics["musicality"]["beat_alignment_pct"]

    print(f"\n  RESULTS:")
    print(f"    μ = {mu:.4f}  ({interp})")
    print(f"    τ* = {tau:.1f} ms")
    print(f"    Beat alignment = {beat_pct:.1f}%")
    print(f"    Flow score = {flow:.1f}")
    print(f"    Stage coverage = {coverage:.2f} m²")
    print(f"    Freezes = {metrics['complexity']['freeze_count']}")
    print(f"    Inversions = {metrics['complexity']['inversion_count']}")
    print(f"    Elapsed = {elapsed:.1f}s")
    print(f"    Output: {out_dir}")

    return {
        "experiment_id": exp_id,
        "title": title,
        "output_dir": str(out_dir),
        "mu": mu,
        "tau_star_ms": tau,
        "beat_alignment_pct": beat_pct,
        "interpretation": interp,
        "flow_score": flow,
        "stage_coverage_m2": coverage,
        "freeze_count": metrics["complexity"]["freeze_count"],
        "inversion_count": metrics["complexity"]["inversion_count"],
        "wall_time_s": round(elapsed, 2),
        "metrics": metrics_clean,
        "meta": meta,
    }


# Simple default pose for synthetic_default source
SMPL_DEFAULT_POSE = [[0, 0.9, 0]] * 22


def _clean_for_json(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── Experiment configurations ───────────────────────────────────────────────

EXPERIMENTS = {
    "EXP-001": {
        "experiment_id": "EXP-001",
        "title": "Synthetic Baseline",
        "joint_source": "synthetic_default",
        "eval_beat_source": "synthetic_120bpm",
        "duration_s": 30.0,
    },
    "EXP-002": {
        "experiment_id": "EXP-002",
        "title": "Toprock On-Beat (lil g)",
        "joint_source": "toprock_onbeat",
        "video_id": "RS0mFARO1x4",
        "seq_idx": 4,
        "eval_beat_source": "brace",
    },
    "EXP-003": {
        "experiment_id": "EXP-003",
        "title": "Toprock Off-Beat Control",
        "joint_source": "toprock_onbeat",  # SAME joints as EXP-002
        "video_id": "RS0mFARO1x4",
        "seq_idx": 4,
        "eval_beat_source": "shifted",     # but evaluated against SHIFTED beats
        "eval_beat_shift_s": 0.24,         # half beat at 125 BPM
    },
    "EXP-004": {
        "experiment_id": "EXP-004",
        "title": "Random Phase Control",
        "joint_source": "toprock_onbeat",  # SAME joints as EXP-002
        "video_id": "RS0mFARO1x4",
        "seq_idx": 4,
        "eval_beat_source": "random",      # but evaluated against RANDOM beats
    },
    "EXP-004b": {
        "experiment_id": "EXP-004b",
        "title": "Random Motion Control",
        "joint_source": "random_control",  # genuinely random motion
        "video_id": "RS0mFARO1x4",
        "seq_idx": 4,
        "eval_beat_source": "brace",       # evaluated against BRACE beats
    },
    "EXP-005a": {
        "experiment_id": "EXP-005a",
        "title": "Cross-Video Neguin",
        "joint_source": "toprock_onbeat",
        "video_id": "HQbI8aWRU7o",
        "seq_idx": 3,
        "eval_beat_source": "brace",
    },
    "EXP-005b": {
        "experiment_id": "EXP-005b",
        "title": "Cross-Video Morris",
        "joint_source": "toprock_onbeat",
        "video_id": "k1RTNQxNt6Q",
        "seq_idx": 1,
        "eval_beat_source": "brace",
    },
    "EXP-006": {
        "experiment_id": "EXP-006",
        "title": "Powermove Stress Test",
        "joint_source": "powermove",
        "video_id": "RS0mFARO1x4",
        "seq_idx": 6,
        "eval_beat_source": "brace",
    },
}


def run_all():
    """Run all experiments and return collected results."""
    results = {}
    order = ["EXP-001", "EXP-002", "EXP-003", "EXP-004", "EXP-004b",
             "EXP-005a", "EXP-005b", "EXP-006"]

    for exp_id in order:
        config = EXPERIMENTS[exp_id]
        result = run_experiment(config)
        results[exp_id] = result

    # EXP-007: SG window sensitivity sweep
    print(f"\n{'='*60}")
    print(f"  EXP-007: SG Window Sensitivity Sweep")
    print(f"{'='*60}")

    sweep_results = {}
    base_config = EXPERIMENTS["EXP-002"].copy()
    for window in [11, 15, 21, 31, 41, 61]:
        cfg = base_config.copy()
        cfg["experiment_id"] = f"EXP-007-w{window}"
        cfg["title"] = f"SG Sweep w={window}"
        cfg["sg_window"] = window
        r = run_experiment(cfg)
        sweep_results[window] = {
            "mu": r["mu"],
            "tau_star_ms": r["tau_star_ms"],
            "beat_alignment_pct": r["beat_alignment_pct"],
        }
        print(f"    w={window}: μ={r['mu']:.4f}, τ*={r['tau_star_ms']:.1f}ms")

    sweep_dir = Path(__file__).parent / "results" / "EXP-007_sg_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with open(sweep_dir / "sweep_results.json", "w") as f:
        json.dump(sweep_results, f, indent=2)
    results["EXP-007"] = {"sweep": sweep_results}

    # ── Summary table ───────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"  {'ID':<10} {'Title':<30} {'μ':>8} {'τ* (ms)':>8} {'Beat%':>6}")
    print(f"  {'-'*10} {'-'*30} {'-'*8} {'-'*8} {'-'*6}")

    for exp_id in order:
        r = results[exp_id]
        print(f"  {exp_id:<10} {r['title']:<30} {r['mu']:>8.4f} {r['tau_star_ms']:>8.1f} {r['beat_alignment_pct']:>5.1f}%")

    # Save master summary
    summary_path = Path(__file__).parent / "results" / "experiment_summary.json"
    summary = {}
    for exp_id in order:
        r = results[exp_id]
        summary[exp_id] = {
            "title": r["title"],
            "mu": r["mu"],
            "tau_star_ms": r["tau_star_ms"],
            "beat_alignment_pct": r["beat_alignment_pct"],
            "interpretation": r["interpretation"],
            "flow_score": r["flow_score"],
            "stage_coverage_m2": r["stage_coverage_m2"],
            "freeze_count": r["freeze_count"],
            "inversion_count": r["inversion_count"],
        }
    summary["EXP-007_sweep"] = sweep_results

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    return results


if __name__ == "__main__":
    run_all()
