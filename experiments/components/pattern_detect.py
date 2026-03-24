"""PatternPanel: windowed autocorrelation + cycle detection visualization."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .panel import Panel, FontCache, C_CYCLIC, TXT, TXT_DIM, BG


class PatternPanel(Panel):
    """Live autocorrelation display with cycle detection."""

    def __init__(self, width: int, height: int, state):
        super().__init__(width, height, state)

    def draw(self, frame_idx: int) -> Image.Image:
        img, d = self._blank()
        f = FontCache.get
        ws = self.state

        # Title
        d.text((10, 4), "PATTERN DETECTION", fill=TXT_DIM, font=f(bold=True, size=13))

        if frame_idx >= ws.frames:
            return img

        # ── Cyclic score timeline (top half) ──────────────────────
        timeline_y = 25
        timeline_h = (self.h - 30) // 2 - 10
        d.rectangle([10, timeline_y, self.w - 10, timeline_y + timeline_h],
                    fill=(28, 28, 38), outline=(50, 50, 60))

        # Draw cyclic score as filled area
        F = ws.frames
        plot_w = self.w - 20
        for x in range(plot_w):
            fi = int(x / plot_w * F)
            fi = min(fi, F - 1)
            val = float(ws.cyclic_score[fi])
            bar_h = int(val * timeline_h)
            if bar_h > 0:
                # Green intensity based on score
                g = int(100 + 155 * val)
                d.line([(10 + x, timeline_y + timeline_h - bar_h),
                        (10 + x, timeline_y + timeline_h)],
                       fill=(60, g, 100))

        # Cyclic regions (highlight bands)
        for region in ws.cyclic_regions:
            rx1 = 10 + int(region["start_frame"] / F * plot_w)
            rx2 = 10 + int(region["end_frame"] / F * plot_w)
            d.rectangle([rx1, timeline_y, rx2, timeline_y + 3], fill=C_CYCLIC)

        # Threshold line
        thresh_y = timeline_y + timeline_h - int(0.5 * timeline_h)
        d.line([(10, thresh_y), (self.w - 10, thresh_y)],
               fill=(255, 200, 50, 80), width=1)
        d.text((self.w - 50, thresh_y - 12), "0.5", fill=TXT_DIM, font=f(size=9))

        # Playhead
        ph_x = 10 + int(frame_idx / F * plot_w)
        d.line([(ph_x, timeline_y), (ph_x, timeline_y + timeline_h)],
               fill=(255, 255, 255), width=2)

        # ── Current state (bottom half) ───────────────────────────
        info_y = timeline_y + timeline_h + 12
        score = float(ws.cyclic_score[frame_idx])

        # Score readout
        d.text((10, info_y), f"P(t) = {score:.3f}",
               fill=C_CYCLIC if score > 0.5 else TXT_DIM,
               font=f(bold=True, size=18))

        # Current region info
        in_region = False
        for region in ws.cyclic_regions:
            if region["start_frame"] <= frame_idx <= region["end_frame"]:
                in_region = True
                dur = region["end_s"] - region["start_s"]
                d.text((10, info_y + 25),
                       f"CYCLIC REGION: {dur:.1f}s  score={region['mean_score']:.3f}",
                       fill=C_CYCLIC, font=f(bold=True, size=13))
                break

        if not in_region:
            d.text((10, info_y + 25), "no cycle detected",
                   fill=TXT_DIM, font=f(size=13))

        # Dominant frequency
        if ws.dominant_freq_hz > 0:
            period = 1.0 / ws.dominant_freq_hz
            d.text((10, info_y + 45),
                   f"dom freq: {ws.dominant_freq_hz:.2f} Hz ({period:.2f}s)",
                   fill=TXT_DIM, font=f(size=12))

        # Cycle count
        d.text((10, info_y + 62),
               f"total cyclic regions: {len(ws.cyclic_regions)}",
               fill=TXT_DIM, font=f(size=12))

        # Region count / duration summary
        if ws.cyclic_regions:
            total_cyclic_s = sum(r["end_s"] - r["start_s"] for r in ws.cyclic_regions)
            total_s = ws.frames / ws.fps
            pct = total_cyclic_s / total_s * 100
            d.text((10, info_y + 78),
                   f"{total_cyclic_s:.1f}s cyclic ({pct:.0f}% of set)",
                   fill=TXT_DIM, font=f(size=12))

        return img
