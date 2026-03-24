"""
JOSH vs GVHMR Comparison Harness

Loads joint outputs from both models on the same video clip and computes:
  1. Procrustes alignment on shared toprock frames
  2. Per-frame MPJPE (Mean Per-Joint Position Error)
  3. Inversion test (headspin/powermove: head below pelvis)
  4. Identity tracking test (max root displacement between frames)
  5. Musicality comparison (mu from cross-correlation)
  6. Stage bounds test

Usage:
  python compare_josh_gvhmr.py \
    --gvhmr-joints results/joints_3d_gvhmr_seq2.npy \
    --josh-joints results/joints_3d_josh_seq2.npy \
    --video data/brace/videos/je265bdPIEU_seq2_audio.mp4 \
    --fps 25
"""

import argparse
import json
import os
import sys
import numpy as np

# Add parent directory to path for importing analyze.py functions
sys.path.insert(0, os.path.dirname(__file__))

try:
    from analyze import compute_movement_signal, compute_musicality, compute_audio_signal
    HAS_ANALYZE = True
except ImportError:
    HAS_ANALYZE = False
    print("WARNING: Could not import from analyze.py. Musicality comparison skipped.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]


def procrustes_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Align source to target using Procrustes analysis (rotation + translation, no scaling).
    Input: (N, 3) point clouds
    Returns: aligned source, error, rotation matrix, translation
    """
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    s_centered = source - mu_s
    t_centered = target - mu_t

    H = s_centered.T @ t_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, d])
    R = Vt.T @ sign_matrix @ U.T
    t = mu_t - R @ mu_s

    aligned = (R @ source.T).T + t
    error = np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=-1)))
    return aligned, error, R, t


def align_sequences(source_seq: np.ndarray, target_seq: np.ndarray,
                    align_frames: slice = slice(0, 100)) -> np.ndarray:
    """
    Align source sequence to target using Procrustes on a reference frame range.
    Input: (F, J, 3) sequences
    Returns: aligned source sequence (F, J, 3)
    """
    # Use mean pose over alignment frames for robust estimation
    src_ref = source_seq[align_frames].reshape(-1, 3)
    tgt_ref = target_seq[align_frames].reshape(-1, 3)

    _, _, R, t = procrustes_align(src_ref, tgt_ref)

    # Apply to all frames
    F, J, _ = source_seq.shape
    aligned = np.zeros_like(source_seq)
    for f in range(F):
        aligned[f] = (R @ source_seq[f].T).T + t
    return aligned


def compute_mpjpe(joints_a: np.ndarray, joints_b: np.ndarray) -> np.ndarray:
    """Per-frame Mean Per-Joint Position Error in mm."""
    errors = np.linalg.norm(joints_a - joints_b, axis=-1)  # (F, J)
    return errors.mean(axis=1) * 1000  # (F,) in mm


def inversion_test(joints: np.ndarray, model_name: str, y_down: bool = False) -> dict:
    """Test whether the model detects inversions (head below pelvis)."""
    head_y = joints[:, 15, 1]   # head joint index 15
    pelvis_y = joints[:, 0, 1]  # pelvis joint index 0
    inverted = head_y > pelvis_y if y_down else head_y < pelvis_y

    return {
        "model": model_name,
        "total_frames": int(len(joints)),
        "inverted_frames": int(np.sum(inverted)),
        "inverted_fraction": round(float(np.mean(inverted)), 3),
        "head_y_mean": round(float(np.mean(head_y)), 3),
        "head_y_std": round(float(np.std(head_y)), 3),
        "pelvis_y_mean": round(float(np.mean(pelvis_y)), 3),
        "head_y_min": round(float(np.min(head_y)), 3),
        "head_y_max": round(float(np.max(head_y)), 3),
    }


def identity_tracking_test(joints: np.ndarray, model_name: str) -> dict:
    """Test for identity swaps via root displacement."""
    root = joints[:, 0, :]  # pelvis trajectory
    displacements = np.linalg.norm(np.diff(root, axis=0), axis=-1)
    max_jump = float(np.max(displacements))
    jump_frame = int(np.argmax(displacements))
    mean_disp = float(np.mean(displacements))

    # Flag frames with large jumps (> 0.3m)
    big_jumps = np.where(displacements > 0.3)[0].tolist()

    return {
        "model": model_name,
        "max_displacement_m": round(max_jump, 4),
        "max_displacement_frame": jump_frame,
        "mean_displacement_m": round(mean_disp, 4),
        "verdict": "PASS" if max_jump < 0.3 else "FAIL",
        "big_jump_frames": big_jumps,
        "root_range_x_m": round(float(root[:, 0].ptp()), 2),
        "root_range_y_m": round(float(root[:, 1].ptp()), 2),
        "root_range_z_m": round(float(root[:, 2].ptp()), 2),
    }


def stage_bounds_test(joints: np.ndarray, model_name: str) -> dict:
    """Check if trajectory stays within reasonable stage bounds."""
    root = joints[:, 0, :]
    x_range = float(root[:, 0].ptp())
    y_range = float(root[:, 1].ptp())
    z_range = float(root[:, 2].ptp())
    reasonable = x_range < 10 and z_range < 10

    return {
        "model": model_name,
        "x_range_m": round(x_range, 2),
        "y_range_m": round(y_range, 2),
        "z_range_m": round(z_range, 2),
        "verdict": "PASS" if reasonable else "FAIL — trajectory escaped stage bounds",
    }


def main():
    parser = argparse.ArgumentParser(description="JOSH vs GVHMR Comparison")
    parser.add_argument("--gvhmr-joints", required=True, help="GVHMR joints_3d.npy path")
    parser.add_argument("--josh-joints", required=True, help="JOSH joints_3d.npy path")
    parser.add_argument("--video", default=None, help="Video with audio (for musicality)")
    parser.add_argument("--fps", type=int, default=25, help="Video FPS")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--no-align", action="store_true", help="Skip Procrustes alignment")
    parser.add_argument("--align-frames", type=int, default=100,
                        help="Number of initial frames to use for alignment")
    args = parser.parse_args()

    output_dir = args.output or os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
    os.makedirs(output_dir, exist_ok=True)

    print("╔══════════════════════════════════════════╗")
    print("║  JOSH vs GVHMR Comparison                ║")
    print("╚══════════════════════════════════════════╝")

    # ─── Load joints ─────────────────────────────────────────
    print("\n▸ Loading joint data...")
    gvhmr = np.load(args.gvhmr_joints)
    josh = np.load(args.josh_joints)
    print(f"  GVHMR: {gvhmr.shape}")
    print(f"  JOSH:  {josh.shape}")

    # Truncate to common length
    n_frames = min(gvhmr.shape[0], josh.shape[0])
    n_joints = min(gvhmr.shape[1], josh.shape[1], 22)
    gvhmr = gvhmr[:n_frames, :n_joints, :]
    josh = josh[:n_frames, :n_joints, :]
    print(f"  Common: {n_frames} frames, {n_joints} joints")

    # ─── Procrustes alignment ────────────────────────────────
    if not args.no_align:
        print(f"\n▸ Procrustes alignment on first {args.align_frames} frames...")
        align_n = min(args.align_frames, n_frames)
        josh_aligned = align_sequences(josh, gvhmr, align_frames=slice(0, align_n))
        _, align_err, _, _ = procrustes_align(
            josh[:align_n].reshape(-1, 3), gvhmr[:align_n].reshape(-1, 3)
        )
        print(f"  Alignment RMSE: {align_err * 1000:.1f}mm")
    else:
        josh_aligned = josh
        align_err = None

    # ─── MPJPE ───────────────────────────────────────────────
    print("\n▸ Computing per-frame MPJPE...")
    mpjpe = compute_mpjpe(gvhmr, josh_aligned)
    print(f"  Mean MPJPE: {np.mean(mpjpe):.1f}mm")
    print(f"  Median MPJPE: {np.median(mpjpe):.1f}mm")
    print(f"  Max MPJPE: {np.max(mpjpe):.1f}mm (frame {np.argmax(mpjpe)})")

    # ─── Inversion test ──────────────────────────────────────
    print("\n▸ Inversion test (head below pelvis)...")
    inv_gvhmr = inversion_test(gvhmr, "GVHMR")
    # TODO: verify y_down — TRAM pred_trans is Y-up; JOSH SMPL output may be too
    inv_josh = inversion_test(josh, "JOSH", y_down=True)
    print(f"  GVHMR: {inv_gvhmr['inverted_frames']}/{inv_gvhmr['total_frames']} "
          f"({inv_gvhmr['inverted_fraction']*100:.1f}%) inverted")
    print(f"  JOSH:  {inv_josh['inverted_frames']}/{inv_josh['total_frames']} "
          f"({inv_josh['inverted_fraction']*100:.1f}%) inverted")

    # ─── Identity tracking ───────────────────────────────────
    print("\n▸ Identity tracking test...")
    id_gvhmr = identity_tracking_test(gvhmr, "GVHMR")
    id_josh = identity_tracking_test(josh, "JOSH")
    print(f"  GVHMR: max jump {id_gvhmr['max_displacement_m']:.3f}m "
          f"at frame {id_gvhmr['max_displacement_frame']} — {id_gvhmr['verdict']}")
    print(f"  JOSH:  max jump {id_josh['max_displacement_m']:.3f}m "
          f"at frame {id_josh['max_displacement_frame']} — {id_josh['verdict']}")

    # ─── Stage bounds ────────────────────────────────────────
    print("\n▸ Stage bounds test...")
    bounds_gvhmr = stage_bounds_test(gvhmr, "GVHMR")
    bounds_josh = stage_bounds_test(josh, "JOSH")
    print(f"  GVHMR: x={bounds_gvhmr['x_range_m']}m, y={bounds_gvhmr['y_range_m']}m, "
          f"z={bounds_gvhmr['z_range_m']}m — {bounds_gvhmr['verdict']}")
    print(f"  JOSH:  x={bounds_josh['x_range_m']}m, y={bounds_josh['y_range_m']}m, "
          f"z={bounds_josh['z_range_m']}m — {bounds_josh['verdict']}")

    # ─── Musicality ──────────────────────────────────────────
    mu_gvhmr = None
    mu_josh = None
    if HAS_ANALYZE and args.video:
        print("\n▸ Musicality comparison...")
        M_gvhmr, _ = compute_movement_signal(gvhmr, fps=args.fps)
        M_josh, _ = compute_movement_signal(josh, fps=args.fps)

        H, beat_times = compute_audio_signal(args.video, len(M_gvhmr), fps=args.fps)

        result_gvhmr = compute_musicality(M_gvhmr, H, fps=args.fps)
        result_josh = compute_musicality(M_josh, H, fps=args.fps)

        mu_gvhmr = result_gvhmr["mu"]
        mu_josh = result_josh["mu"]

        print(f"  GVHMR: μ = {mu_gvhmr:.3f}, τ* = {result_gvhmr['tau_star_ms']:.0f}ms")
        print(f"  JOSH:  μ = {mu_josh:.3f}, τ* = {result_josh['tau_star_ms']:.0f}ms")
    elif not args.video:
        print("\n  Musicality skipped (no --video provided)")

    # ─── Summary ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<30} {'GVHMR':>12} {'JOSH':>12}")
    print(f"  {'-' * 54}")
    print(f"  {'Inverted frames':<30} {inv_gvhmr['inverted_fraction']*100:>11.1f}% {inv_josh['inverted_fraction']*100:>11.1f}%")
    print(f"  {'Max root jump (m)':<30} {id_gvhmr['max_displacement_m']:>12.3f} {id_josh['max_displacement_m']:>12.3f}")
    print(f"  {'Identity tracking':<30} {id_gvhmr['verdict']:>12} {id_josh['verdict']:>12}")
    print(f"  {'Stage bounds':<30} {bounds_gvhmr['verdict']:>12} {bounds_josh['verdict']:>12}")
    if mu_gvhmr is not None:
        print(f"  {'Musicality μ':<30} {mu_gvhmr:>12.3f} {mu_josh:>12.3f}")
    print(f"  {'Mean MPJPE (mm)':<30} {'—':>12} {np.mean(mpjpe):>12.1f}")
    print(f"{'=' * 60}")

    # ─── Save results ────────────────────────────────────────
    comparison = {
        "models": ["GVHMR", "JOSH"],
        "n_frames": n_frames,
        "n_joints": n_joints,
        "fps": args.fps,
        "alignment": {
            "method": "procrustes" if not args.no_align else "none",
            "reference_frames": args.align_frames,
            "rmse_mm": round(align_err * 1000, 1) if align_err else None,
        },
        "mpjpe": {
            "mean_mm": round(float(np.mean(mpjpe)), 1),
            "median_mm": round(float(np.median(mpjpe)), 1),
            "max_mm": round(float(np.max(mpjpe)), 1),
            "max_frame": int(np.argmax(mpjpe)),
        },
        "inversion_test": {"GVHMR": inv_gvhmr, "JOSH": inv_josh},
        "identity_tracking": {"GVHMR": id_gvhmr, "JOSH": id_josh},
        "stage_bounds": {"GVHMR": bounds_gvhmr, "JOSH": bounds_josh},
        "musicality": {
            "GVHMR": {"mu": mu_gvhmr} if mu_gvhmr else None,
            "JOSH": {"mu": mu_josh} if mu_josh else None,
        },
    }

    out_path = os.path.join(output_dir, "josh_vs_gvhmr_comparison.json")
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # ─── Plots ───────────────────────────────────────────────
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        t = np.arange(n_frames) / args.fps

        # Plot 1: MPJPE over time
        axes[0].plot(t, mpjpe, "r-", linewidth=0.8, alpha=0.8)
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("MPJPE (mm)")
        axes[0].set_title(f"GVHMR vs JOSH — Per-Frame MPJPE (mean={np.mean(mpjpe):.1f}mm)")
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Head height comparison
        axes[1].plot(t, gvhmr[:, 15, 1], label="GVHMR head_y", alpha=0.8, linewidth=0.8)
        axes[1].plot(t, josh[:, 15, 1], label="JOSH head_y", alpha=0.8, linewidth=0.8)
        axes[1].plot(t, gvhmr[:, 0, 1], "--", label="GVHMR pelvis_y", alpha=0.5, linewidth=0.6)
        axes[1].plot(t, josh[:, 0, 1], "--", label="JOSH pelvis_y", alpha=0.5, linewidth=0.6)
        axes[1].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Height (m)")
        axes[1].set_title("Head vs Pelvis Height — Inversion Detection")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Root trajectory (top-down view)
        axes[2].plot(gvhmr[:, 0, 0], gvhmr[:, 0, 2], "b-", label="GVHMR", alpha=0.6, linewidth=0.8)
        axes[2].plot(josh[:, 0, 0], josh[:, 0, 2], "r-", label="JOSH", alpha=0.6, linewidth=0.8)
        axes[2].plot(gvhmr[0, 0, 0], gvhmr[0, 0, 2], "bo", markersize=8, label="Start (GVHMR)")
        axes[2].plot(josh[0, 0, 0], josh[0, 0, 2], "ro", markersize=8, label="Start (JOSH)")
        axes[2].set_xlabel("X (m)")
        axes[2].set_ylabel("Z (m)")
        axes[2].set_title("Root Trajectory — Top-Down View")
        axes[2].legend(fontsize=8)
        axes[2].set_aspect("equal")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "josh_vs_gvhmr_comparison.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved: {plot_path}")


if __name__ == "__main__":
    main()
