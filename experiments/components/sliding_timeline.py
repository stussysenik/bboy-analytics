"""SlidingTimeline: DJ-style scrolling waveform with center-fixed playhead.

Three audio frequency bands (bass/perc/harm) scroll right-to-left past a
fixed playhead. Beat dots track hit/miss timing. Inspired by Serato and
rekordbox waveform displays.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .panel import Panel, FontCache, BG, TXT, TXT_DIM, C_BASS, C_PERC, C_HARM

# Dimming factor for audio that has already passed the playhead
PAST_DIM = 0.7


class SlidingTimeline(Panel):
    """Scrolling audio timeline with center-fixed playhead."""

    def __init__(self, width: int, height: int, state,
                 perc: np.ndarray | None = None,
                 harm: np.ndarray | None = None,
                 bass: np.ndarray | None = None,
                 window_seconds: float = 8.0):
        super().__init__(width, height, state)
        self.perc_raw = perc
        self.harm_raw = harm
        self.bass_raw = bass
        self.window_s = window_seconds

        # Populated by prerender()
        self._bands: list[dict] = []        # active bands after None filtering
        self._beat_data: list[dict] = []     # beat entries within clip duration
        self._total_dur: float = 0.0
        self._has_audio = False

    # ── prerender: normalize once ────────────────────────────────────

    def prerender(self) -> None:
        ws = self.state
        self._total_dur = ws.frames / ws.fps

        # Build active band list (skip None arrays)
        raw_bands = [
            ("BASS", self.bass_raw, C_BASS),
            ("PERC", self.perc_raw, C_PERC),
            ("HARM", self.harm_raw, C_HARM),
        ]
        self._bands = []
        for label, arr, color in raw_bands:
            if arr is not None and len(arr) > 0:
                normed = arr.astype(np.float64)
                mx = normed.max()
                if mx > 1e-8:
                    normed /= mx
                self._bands.append({
                    "label": label,
                    "data": normed,
                    "color": color,
                    "dur": self._total_dur,  # assume arrays span full clip
                })

        self._has_audio = len(self._bands) > 0

        # Pre-sort beat data by time for efficient windowed lookup
        if ws.beat_hits:
            self._beat_data = sorted(ws.beat_hits, key=lambda b: b["time_s"])

    # ── draw: per-frame rendering ────────────────────────────────────

    def draw(self, frame_idx: int) -> Image.Image:
        img, d = self._blank()
        ws = self.state
        f = FontCache.get

        # Fallback when no audio data at all
        if not self._has_audio:
            d.text((self.w // 2 - 40, self.h // 2 - 8),
                   "NO AUDIO", fill=TXT_DIM, font=f(bold=True, size=18))
            return img

        current_t = frame_idx / ws.fps
        half_win = self.window_s / 2.0
        win_start = current_t - half_win  # time at left edge
        win_end = current_t + half_win    # time at right edge
        playhead_x = self.w // 2

        # Divide vertical space among active bands + beats row
        n_rows = len(self._bands) + 1  # +1 for beat dots
        row_h = self.h // n_rows

        # ── Draw each audio band ─────────────────────────────────────
        for i, band in enumerate(self._bands):
            y_top = i * row_h
            y_bot = y_top + row_h - 1
            band_h = row_h - 2  # padding
            self._draw_band(d, band, win_start, win_end, current_t,
                            playhead_x, y_top, band_h)
            # Row label
            d.text((4, y_top + 2), band["label"],
                   fill=band["color"], font=f(bold=True, size=11))

        # ── Draw beat dots ───────────────────────────────────────────
        beats_y_top = len(self._bands) * row_h
        beats_cy = beats_y_top + row_h // 2

        for bh in self._beat_data:
            bt = bh["time_s"]
            if bt < win_start or bt > win_end:
                continue
            bx = self._time_to_x(bt, win_start, win_end)
            is_hit = bh.get("hit", False)
            if is_hit:
                r = 5
                fill = (80, 255, 100)
            else:
                r = 3
                fill = (255, 70, 70)
            d.ellipse([bx - r, beats_cy - r, bx + r, beats_cy + r], fill=fill)

        # Label for beats row
        d.text((4, beats_y_top + 2), "BEATS",
               fill=TXT_DIM, font=f(bold=True, size=11))

        # ── Playhead: white vertical line + triangle marker ──────────
        d.line([(playhead_x, 0), (playhead_x, self.h)],
               fill=(255, 255, 255), width=2)
        # Downward triangle at top
        tri_size = 6
        d.polygon([
            (playhead_x - tri_size, 0),
            (playhead_x + tri_size, 0),
            (playhead_x, tri_size * 2),
        ], fill=(255, 255, 255))

        return img

    # ── helpers ──────────────────────────────────────────────────────

    def _time_to_x(self, t: float, win_start: float, win_end: float) -> int:
        """Map a time value to a pixel x position within the visible window."""
        frac = (t - win_start) / (win_end - win_start)
        return int(frac * self.w)

    def _draw_band(self, d: ImageDraw.ImageDraw, band: dict,
                   win_start: float, win_end: float, current_t: float,
                   playhead_x: int, y_top: int, band_h: int) -> None:
        """Draw a single audio frequency band as vertical bars."""
        data = band["data"]
        color = band["color"]
        n_samples = len(data)
        baseline = y_top + band_h  # bars grow upward from baseline

        cr, cg, cb = color

        for px in range(self.w):
            # Map pixel to time
            t = win_start + (px / self.w) * (win_end - win_start)

            # Skip pixels outside the clip duration
            if t < 0 or t >= self._total_dur:
                continue

            # Map time to sample index in the band array
            sample_idx = int(t / self._total_dur * n_samples)
            sample_idx = min(sample_idx, n_samples - 1)
            amp = float(data[sample_idx])

            # Dim past audio (left of playhead)
            dim = PAST_DIM if px < playhead_x else 1.0

            # Color intensity scales with amplitude
            intensity = 0.4 + 0.6 * amp  # base brightness + amplitude modulation
            scale = dim * intensity

            r = min(255, int(cr * scale))
            g = min(255, int(cg * scale))
            b = min(255, int(cb * scale))

            bar_h = max(1, int(amp * band_h * 0.85))
            d.line([(px, baseline - bar_h), (px, baseline)], fill=(r, g, b))
