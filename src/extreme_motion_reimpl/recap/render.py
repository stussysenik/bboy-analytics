"""Render battle recap visualizations: static PNGs and composited MP4."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path


def render_energy_flow(metrics: dict, output_path: Path, fps: float) -> None:
    """Render energy + musicality timeline as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    energy = metrics["energy"]["energy_curve"]
    t = np.arange(len(energy)) / fps

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Energy curve
    axes[0].plot(t, energy, color="#ff6b35", linewidth=0.8, alpha=0.9)
    axes[0].fill_between(t, energy, alpha=0.2, color="#ff6b35")
    axes[0].set_ylabel("Energy")
    axes[0].set_title("Energy Flow Over Set")
    axes[0].grid(True, alpha=0.2)

    # Freeze events
    for freeze in metrics["complexity"]["freeze_events"]:
        axes[0].axvspan(freeze["start_s"], freeze["end_s"], color="blue", alpha=0.15)

    # Peak moments
    for peak in metrics["energy"]["peak_moments"]:
        axes[0].axvline(peak["time_s"], color="red", alpha=0.4, linewidth=0.5)

    # Flow score (jerk) if available
    mu = metrics["musicality"]["mu"]
    axes[1].set_ylabel("Flow Score")
    axes[1].set_xlabel("Time (s)")
    axes[1].text(0.02, 0.95, f"μ = {mu:.3f}  |  Flow = {metrics['flow']['flow_score']:.0f}",
                 transform=axes[1].transAxes, fontsize=11, va="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def render_spatial_heatmap(metrics: dict, output_path: Path) -> None:
    """Render top-down stage coverage heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj = np.array(metrics["space"]["com_trajectory"])
    if traj.ndim != 2 or traj.shape[1] != 2:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.hist2d(traj[:, 0], traj[:, 1], bins=30, cmap="hot")
    ax.plot(traj[:, 0], traj[:, 1], color="white", alpha=0.3, linewidth=0.5)
    ax.plot(traj[0, 0], traj[0, 1], "go", markersize=8, label="Start")
    ax.plot(traj[-1, 0], traj[-1, 1], "rs", markersize=8, label="End")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"Stage Coverage — {metrics['space']['stage_coverage_m2']:.1f} m²")
    ax.set_aspect("equal")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def render_com_trajectory(metrics: dict, output_path: Path, fps: float) -> None:
    """Render COM path + vertical (Y) over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj = np.array(metrics["space"]["com_trajectory"])
    y = np.array(metrics["space"]["com_y"])
    t = np.arange(len(y)) / fps

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top-down path
    axes[0].plot(traj[:, 0], traj[:, 1], linewidth=0.8, alpha=0.7)
    axes[0].plot(traj[0, 0], traj[0, 1], "go", markersize=8)
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Z (m)")
    axes[0].set_title("COM Path (top-down)")
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.2)

    # Vertical
    axes[1].plot(t, y, linewidth=0.8, color="#2196F3")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Height (m)")
    axes[1].set_title(f"Vertical — range {metrics['space']['vertical_range_m']:.2f}m")
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def render_all(metrics: dict, output_dir: Path, fps: float) -> list[str]:
    """Render all static visualizations. Returns list of created files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []

    try:
        import matplotlib
    except ImportError:
        print("  WARNING: matplotlib not installed, skipping visualizations")
        return created

    render_energy_flow(metrics, output_dir / "energy_flow.png", fps)
    created.append("energy_flow.png")

    render_spatial_heatmap(metrics, output_dir / "spatial_heatmap.png")
    created.append("spatial_heatmap.png")

    render_com_trajectory(metrics, output_dir / "com_trajectory.png", fps)
    created.append("com_trajectory.png")

    return created
