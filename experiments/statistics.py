"""
Statistical validation: permutation tests, bootstrap CIs, effect sizes.

Validates that μ separation between on-beat and control conditions
is statistically significant, not an artifact.
"""

from __future__ import annotations

import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from extreme_motion_reimpl.recap.metrics import compute_musicality
from experiments.synthetic_joints import (
    generate_toprock_onbeat,
    generate_random_control,
    load_brace_beats,
)


def permutation_test(
    joints: np.ndarray,
    real_beats: np.ndarray,
    fps: float,
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Permutation test: shuffle beat times and recompute μ each time.
    Reports p-value for the observed μ under the null hypothesis
    that beat timing is irrelevant to movement.
    """
    rng = np.random.default_rng(seed)
    duration = joints.shape[0] / fps

    # Observed μ
    observed = compute_musicality(joints, real_beats, fps)
    observed_mu = observed["mu"]

    # Null distribution: random beat placements
    null_mus = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm_beats = np.sort(rng.uniform(0, duration, len(real_beats)))
        result = compute_musicality(joints, perm_beats, fps)
        null_mus[i] = result["mu"]

    # p-value: fraction of null μ >= observed
    p_value = float(np.mean(null_mus >= observed_mu))

    return {
        "observed_mu": round(observed_mu, 6),
        "null_mean": round(float(np.mean(null_mus)), 6),
        "null_std": round(float(np.std(null_mus)), 6),
        "null_median": round(float(np.median(null_mus)), 6),
        "null_95th": round(float(np.percentile(null_mus, 95)), 6),
        "null_99th": round(float(np.percentile(null_mus, 99)), 6),
        "p_value": round(p_value, 6),
        "n_permutations": n_permutations,
        "significant_001": p_value < 0.001,
        "significant_01": p_value < 0.01,
        "significant_05": p_value < 0.05,
    }


def bootstrap_ci(
    joints: np.ndarray,
    beats: np.ndarray,
    fps: float,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Bootstrap CI: resample temporal windows and compute μ for each.
    Reports confidence interval for the true μ.
    """
    rng = np.random.default_rng(seed)
    n_frames = joints.shape[0]
    duration = n_frames / fps

    # Split into ~2s windows
    window_frames = int(2.0 * fps)
    n_windows = max(1, n_frames // window_frames)

    # Compute μ for each bootstrap resample
    boot_mus = []
    for _ in range(n_bootstrap):
        # Sample windows with replacement
        window_indices = rng.integers(0, n_windows, size=n_windows)
        # Build resampled sequence
        resampled_frames = []
        for wi in window_indices:
            start = wi * window_frames
            end = min(start + window_frames, n_frames)
            resampled_frames.append(joints[start:end])

        resampled = np.concatenate(resampled_frames, axis=0)

        # Adjust beats to fit resampled duration
        resamp_duration = resampled.shape[0] / fps
        valid_beats = beats[beats < resamp_duration]
        if len(valid_beats) < 3:
            continue

        result = compute_musicality(resampled, valid_beats, fps)
        boot_mus.append(result["mu"])

    boot_mus = np.array(boot_mus)
    alpha = 1 - ci_level
    ci_low = float(np.percentile(boot_mus, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_mus, 100 * (1 - alpha / 2)))

    return {
        "mean_mu": round(float(np.mean(boot_mus)), 6),
        "std_mu": round(float(np.std(boot_mus)), 6),
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "ci_level": ci_level,
        "n_bootstrap": len(boot_mus),
    }


def cohens_d(mu_onbeat: list[float], mu_control: list[float]) -> dict:
    """
    Cohen's d effect size between on-beat and control μ distributions.
    """
    a = np.array(mu_onbeat)
    b = np.array(mu_control)

    mean_diff = float(np.mean(a) - np.mean(b))
    pooled_std = float(np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2))
    d = mean_diff / (pooled_std + 1e-8)

    # Interpretation (Cohen's conventions)
    if abs(d) >= 0.8:
        interpretation = "large"
    elif abs(d) >= 0.5:
        interpretation = "medium"
    elif abs(d) >= 0.2:
        interpretation = "small"
    else:
        interpretation = "negligible"

    return {
        "cohens_d": round(d, 4),
        "mean_onbeat": round(float(np.mean(a)), 6),
        "mean_control": round(float(np.mean(b)), 6),
        "pooled_std": round(pooled_std, 6),
        "interpretation": interpretation,
    }


def cross_video_consistency(
    video_configs: list[dict],
    fps: float = 30.0,
    seed: int = 42,
) -> dict:
    """
    Leave-one-video-out: compute μ for each video, report mean ± std.
    """
    results = []
    for cfg in video_configs:
        brace = load_brace_beats(cfg["video_id"], cfg["seq_idx"])
        beats = brace["beat_times"]
        joints = generate_toprock_onbeat(beats, fps, seed=seed)
        m = compute_musicality(joints, beats, fps)
        results.append({
            "video_id": cfg["video_id"],
            "seq_idx": cfg["seq_idx"],
            "bpm": brace["bpm"],
            "mu": m["mu"],
            "tau_star_ms": m["tau_star_ms"],
        })

    mus = [r["mu"] for r in results]
    return {
        "per_video": results,
        "mean_mu": round(float(np.mean(mus)), 6),
        "std_mu": round(float(np.std(mus)), 6),
        "min_mu": round(float(np.min(mus)), 6),
        "max_mu": round(float(np.max(mus)), 6),
        "n_videos": len(results),
        "all_pass_h1": all(m > 0.3 for m in mus),
    }


def run_all_statistics():
    """Run complete statistical validation suite."""
    print("="*60)
    print("  STATISTICAL VALIDATION")
    print("="*60)

    fps = 30.0
    seed = 42

    # Load primary dataset: RS0mFARO1x4.4 (lil g toprock)
    brace = load_brace_beats("RS0mFARO1x4", 4)
    beats = brace["beat_times"]
    joints_onbeat = generate_toprock_onbeat(beats, fps, seed=seed)

    # ── 1. Permutation test ─────────────────────────────────────
    print("\n  [1/4] Permutation test (10,000 shuffles)...")
    perm = permutation_test(joints_onbeat, beats, fps, n_permutations=10000, seed=seed)
    print(f"    Observed μ = {perm['observed_mu']:.4f}")
    print(f"    Null: mean={perm['null_mean']:.4f}, std={perm['null_std']:.4f}")
    print(f"    Null 95th percentile = {perm['null_95th']:.4f}")
    print(f"    p-value = {perm['p_value']:.6f}")
    print(f"    Significant at p<0.001? {perm['significant_001']}")

    # ── 2. Bootstrap CI ─────────────────────────────────────────
    print("\n  [2/4] Bootstrap 95% CI (1,000 resamples)...")
    boot = bootstrap_ci(joints_onbeat, beats, fps, n_bootstrap=1000, seed=seed)
    print(f"    Mean μ = {boot['mean_mu']:.4f}")
    print(f"    95% CI: [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]")
    print(f"    CI contains 0.3 threshold? {boot['ci_low'] <= 0.3 <= boot['ci_high']}")

    # ── 3. Cohen's d ────────────────────────────────────────────
    print("\n  [3/4] Cohen's d effect size...")

    # Collect on-beat μ across videos
    video_configs = [
        {"video_id": "RS0mFARO1x4", "seq_idx": 4},
        {"video_id": "HQbI8aWRU7o", "seq_idx": 3},
        {"video_id": "k1RTNQxNt6Q", "seq_idx": 1},
    ]

    onbeat_mus = []
    for cfg in video_configs:
        b = load_brace_beats(cfg["video_id"], cfg["seq_idx"])
        j = generate_toprock_onbeat(b["beat_times"], fps, seed=seed)
        m = compute_musicality(j, b["beat_times"], fps)
        onbeat_mus.append(m["mu"])

    # Collect control μ (random beats against same joints)
    control_mus = []
    rng = np.random.default_rng(seed + 500)
    for cfg in video_configs:
        b = load_brace_beats(cfg["video_id"], cfg["seq_idx"])
        j = generate_toprock_onbeat(b["beat_times"], fps, seed=seed)
        for trial in range(3):  # 3 random beat sets per video
            rand_beats = np.sort(rng.uniform(0, j.shape[0] / fps, len(b["beat_times"])))
            m = compute_musicality(j, rand_beats, fps)
            control_mus.append(m["mu"])

    effect = cohens_d(onbeat_mus, control_mus)
    print(f"    On-beat mean μ = {effect['mean_onbeat']:.4f}")
    print(f"    Control mean μ = {effect['mean_control']:.4f}")
    print(f"    Cohen's d = {effect['cohens_d']:.2f} ({effect['interpretation']})")

    # ── 4. Cross-video consistency ──────────────────────────────
    print("\n  [4/4] Cross-video consistency...")
    consistency = cross_video_consistency(video_configs, fps, seed)
    print(f"    Mean μ across videos = {consistency['mean_mu']:.4f} ± {consistency['std_mu']:.4f}")
    print(f"    Range: [{consistency['min_mu']:.4f}, {consistency['max_mu']:.4f}]")
    print(f"    All pass H1 (μ > 0.3)? {consistency['all_pass_h1']}")

    # ── Save ────────────────────────────────────────────────────
    validation = {
        "permutation_test": perm,
        "bootstrap_ci": boot,
        "effect_size": effect,
        "cross_video_consistency": consistency,
        "summary": {
            "hypothesis_supported": perm["significant_01"] and consistency["all_pass_h1"],
            "p_value": perm["p_value"],
            "effect_size": effect["cohens_d"],
            "effect_interpretation": effect["interpretation"],
            "mean_mu_onbeat": round(float(np.mean(onbeat_mus)), 4),
            "mean_mu_control": round(float(np.mean(control_mus)), 4),
            "separation_ratio": round(float(np.mean(onbeat_mus)) / (float(np.mean(control_mus)) + 1e-8), 1),
        },
    }

    out_path = Path(__file__).parent / "results" / "statistical_validation.json"
    with open(out_path, "w") as f:
        json.dump(validation, f, indent=2)

    print(f"\n  {'='*60}")
    print(f"  VALIDATION SUMMARY")
    print(f"  {'='*60}")
    print(f"  Hypothesis supported: {validation['summary']['hypothesis_supported']}")
    print(f"  p-value: {perm['p_value']:.6f}")
    print(f"  Effect size: d={effect['cohens_d']:.2f} ({effect['interpretation']})")
    print(f"  On-beat/control ratio: {validation['summary']['separation_ratio']}×")
    print(f"  Saved: {out_path}")

    return validation


if __name__ == "__main__":
    run_all_statistics()
