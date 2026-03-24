"""Bottom strip renderer: composite timeline with interacting metrics.

NOT isolated metric graphs. Each row shows INTERACTIONS:
- Beat timeline with cluster-colored hit indicators
- Energy × Musicality superimposed
- Hip-COM magnitude + phase color bar
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .body_state import BodyState
from .color_system import (
    BEAT_HIT_COLOR,
    BEAT_MISS_COLOR,
    CLUSTER_COLORS,
    ENERGY_COLOR,
    GRID_COLOR,
    HIP_COM_COLOR,
    MUSICALITY_COLOR,
    PANEL_BG,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    joint_color,
    phase_color,
)

try:
    _FONT_SM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    _FONT_LEGEND = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    _FONT_SCORE = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
except OSError:
    _FONT_SM = ImageFont.load_default()
    _FONT_LEGEND = ImageFont.load_default()
    _FONT_SCORE = ImageFont.load_default()

ROW_HEIGHT = 60
LEGEND_HEIGHT = 40
MARGIN = 12


def render_timeline_strip(
    all_states: List[BodyState],
    current_idx: int,
    beats: Optional[np.ndarray] = None,
    beat_hits: Optional[List[List[int]]] = None,
    strip_w: int = 1920,
    strip_h: int = 220,
) -> Image.Image:
    """Render composite timeline strip.

    Args:
        all_states: Complete sequence of BodyStates
        current_idx: Current frame index (for playhead)
        beats: Beat timestamps in seconds
        beat_hits: Per-beat list of joint indices that hit
        strip_w: Strip width
        strip_h: Strip height

    Returns:
        PIL Image (RGBA)
    """
    img = Image.new("RGBA", (strip_w, strip_h), PANEL_BG + (255,))
    draw = ImageDraw.Draw(img)

    T = len(all_states)
    if T == 0:
        return img

    total_dur = all_states[-1].timestamp
    tl_x = MARGIN + 60  # leave room for labels
    tl_w = strip_w - tl_x - MARGIN

    def time_to_x(t: float) -> int:
        return int(tl_x + (t / max(total_dur, 0.01)) * tl_w)

    def frame_to_x(f: int) -> int:
        return time_to_x(f / max(T, 1) * total_dur)

    # ── Row 1: Beat timeline (0 to ROW_HEIGHT) ──
    y1 = 0
    draw.text((MARGIN, y1 + 4), "BEATS", fill=TEXT_SECONDARY, font=_FONT_SM)

    # Draw beat dots
    if beats is not None:
        fps = T / max(total_dur, 0.01)
        total_hits = 0
        for i, beat_sec in enumerate(beats):
            bx = time_to_x(beat_sec)
            if bx < tl_x or bx > tl_x + tl_w:
                continue

            hit_joints = beat_hits[i] if beat_hits and i < len(beat_hits) else []
            is_hit = len(hit_joints) > 0
            total_hits += 1 if is_hit else 0

            # Main dot
            color = BEAT_HIT_COLOR if is_hit else BEAT_MISS_COLOR
            r = 5 if is_hit else 3
            by = y1 + ROW_HEIGHT // 2
            draw.ellipse([bx - r, by - r, bx + r, by + r], fill=color)

            # Cluster indicator for hits (tiny colored dots below)
            if is_hit and hit_joints:
                # Show the dominant cluster that hit
                from collections import Counter
                from .color_system import JOINT_CLUSTER
                clusters = [JOINT_CLUSTER.get(j, "torso") for j in hit_joints]
                dominant = Counter(clusters).most_common(1)[0][0]
                cc = CLUSTER_COLORS[dominant]
                draw.ellipse([bx - 2, by + 8, bx + 2, by + 12], fill=cc)

        # Hit rate text
        n_beats = len(beats)
        pct = (total_hits / max(n_beats, 1)) * 100
        draw.text(
            (tl_x + tl_w - 120, y1 + 4),
            f"{total_hits}/{n_beats} ({pct:.1f}%)",
            fill=TEXT_PRIMARY,
            font=_FONT_SM,
        )

    # ── Row 2: Energy × Musicality composite (ROW_HEIGHT to 2*ROW_HEIGHT) ──
    y2 = ROW_HEIGHT
    draw.text((MARGIN, y2 + 4), "E × μ", fill=TEXT_SECONDARY, font=_FONT_SM)

    # Compute normalized energy and musicality curves
    energies = np.array([s.total_energy for s in all_states])
    musicalities = np.array([s.musicality for s in all_states])

    e_max = max(energies.max(), 1e-8)
    m_max = max(musicalities.max(), 1e-8) if musicalities.max() > 0 else 1.0

    # Energy fill (orange, 20% opacity)
    energy_points = []
    for f in range(T):
        x = frame_to_x(f)
        norm_e = energies[f] / e_max
        y = int(y2 + ROW_HEIGHT - norm_e * (ROW_HEIGHT - 10) - 5)
        energy_points.append((x, y))

    if len(energy_points) > 1:
        # Fill under curve
        fill_points = [(energy_points[0][0], y2 + ROW_HEIGHT)]
        fill_points.extend(energy_points)
        fill_points.append((energy_points[-1][0], y2 + ROW_HEIGHT))
        draw.polygon(fill_points, fill=ENERGY_COLOR + (50,))
        # Line
        for i in range(len(energy_points) - 1):
            draw.line([energy_points[i], energy_points[i + 1]],
                      fill=ENERGY_COLOR + (200,), width=1)

    # Musicality line (pink, on same axes)
    if m_max > 0:
        mu_points = []
        for f in range(T):
            x = frame_to_x(f)
            norm_m = musicalities[f] / m_max
            y = int(y2 + ROW_HEIGHT - norm_m * (ROW_HEIGHT - 10) - 5)
            mu_points.append((x, y))

        if len(mu_points) > 1:
            for i in range(len(mu_points) - 1):
                draw.line([mu_points[i], mu_points[i + 1]],
                          fill=MUSICALITY_COLOR + (200,), width=1)

    # ── Row 3: Hip-COM magnitude + phase bar (2*ROW_HEIGHT to 3*ROW_HEIGHT) ──
    y3 = 2 * ROW_HEIGHT
    draw.text((MARGIN, y3 + 4), "HIP→COM", fill=TEXT_SECONDARY, font=_FONT_SM)

    hip_com_mags = np.array([s.hip_com_magnitude for s in all_states])
    hc_max = max(hip_com_mags.max(), 0.01)

    # Phase color bar (bottom half of row)
    phase_bar_y = y3 + ROW_HEIGHT - 14
    prev_phase = all_states[0].phase
    seg_start = 0
    for f in range(T + 1):
        cur_phase = all_states[f].phase if f < T else ""
        if cur_phase != prev_phase or f == T:
            x_start = frame_to_x(seg_start)
            x_end = frame_to_x(f - 1)
            pc = phase_color(prev_phase)
            draw.rectangle(
                [x_start, phase_bar_y, x_end, phase_bar_y + 12],
                fill=pc + (180,),
            )
            seg_start = f
            prev_phase = cur_phase

    # Hip-COM line (gold)
    hc_points = []
    for f in range(T):
        x = frame_to_x(f)
        norm_hc = hip_com_mags[f] / hc_max
        y = int(y3 + (ROW_HEIGHT - 18) - norm_hc * (ROW_HEIGHT - 28) - 5)
        hc_points.append((x, y))

    if len(hc_points) > 1:
        for i in range(len(hc_points) - 1):
            draw.line([hc_points[i], hc_points[i + 1]],
                      fill=HIP_COM_COLOR + (200,), width=1)

    # ── Row 4: Legend + score ──
    y4 = 3 * ROW_HEIGHT
    lx = MARGIN
    for cluster_name, color in CLUSTER_COLORS.items():
        draw.ellipse([lx, y4 + 12, lx + 10, y4 + 22], fill=color)
        draw.text((lx + 14, y4 + 10), cluster_name.upper(), fill=TEXT_PRIMARY, font=_FONT_LEGEND)
        lx += len(cluster_name) * 9 + 36

    # Score readout
    if beats is not None and beat_hits is not None:
        total_hits = sum(1 for h in beat_hits if len(h) > 0)
        n_beats = len(beats)
        pct = (total_hits / max(n_beats, 1)) * 100
        score_text = f"TRIVIUM Beats: {total_hits}/{n_beats} ({pct:.1f}%)"
        draw.text(
            (strip_w - 260, y4 + 10),
            score_text,
            fill=TEXT_PRIMARY,
            font=_FONT_SCORE,
        )

    # ── Playhead (vertical white line across all rows) ──
    px = frame_to_x(current_idx)
    draw.line([(px, 0), (px, 3 * ROW_HEIGHT)], fill=(255, 255, 255, 180), width=1)

    # Grid lines (subtle)
    for sec in range(int(total_dur) + 1):
        gx = time_to_x(float(sec))
        draw.line([(gx, 0), (gx, 3 * ROW_HEIGHT)], fill=GRID_COLOR + (100,), width=1)

    return img
