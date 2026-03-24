"""
Video 4 v3: "The Breakdown" — Multi-view ghost trails + simplified analytics.

Layout (1920x1080):
  Top 540px:     Battle video (full width) + stats badge + move label
  Mid 340px:     3 orthographic views: Front (XY) | Side (ZY) | Top-down (XZ)
  Row 1 (80px):  Move type bar with energy fill
  Row 2 (80px):  Musicality hero stat: X/Y beats hit + dot strip + μ
  Footer (40px): BPM | Duration | Peak velocity | Coverage

Output: experiments/exports/v2/trails.mp4
"""

from __future__ import annotations
import json, subprocess, sys, time, csv, tempfile, shutil
from datetime import datetime
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Layout ────────────────────────────────────────────────────────
W, H = 1920, 1080
FPS = 29.97
VID_H = 540        # Video height
VIEW_H = 340       # Three-view panel height
VIEW_W = W // 3    # Each view width (640)
MOVE_ROW_H = 80    # Move bar + energy
MU_ROW_H = 80      # Musicality hero
FOOTER_H = 40      # Stats footer
# Total: 540 + 340 + 80 + 80 + 40 = 1080

TRAIL_LEN = 60     # 2 seconds
SWAP_THRESH = 0.3

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# ── Colors ────────────────────────────────────────────────────────
BG = (14, 14, 20)
GRID_C = (30, 30, 38)
TXT = (255, 255, 255)
TXT_DIM = (120, 120, 135)
C_COM = (255, 80, 80)
SEG_COLORS = {
    "toprock":   (33, 150, 243),
    "footwork":  (76, 175, 80),
    "powermove": (255, 87, 34),
    "freeze":    (220, 60, 60),
}
C_ENERGY_LO = (50, 130, 200)
C_ENERGY_HI = (255, 80, 40)
C_BEAT_HIT = (50, 220, 100)
C_BEAT_MISS = (60, 60, 70)
C_MU = (180, 100, 255)

CLUSTER_COLORS = {
    "legs":  (0xFF, 0x6B, 0x35),
    "torso": (0xE6, 0x39, 0x46),
    "arms":  (0x45, 0x7B, 0x9D),
    "hands": (0xA8, 0x55, 0xF7),
    "head":  (0xE0, 0xF2, 0xFE),
}
JOINT_CLUSTER = {}
for cluster, indices in {
    "legs":  [1, 2, 4, 5, 7, 8, 10, 11],
    "torso": [0, 3, 6, 9],
    "arms":  [13, 14, 16, 17, 18, 19],
    "hands": [20, 21],
    "head":  [12, 15],
}.items():
    for idx in indices:
        JOINT_CLUSTER[idx] = cluster

BONES = [
    (0,1),(0,2),(0,3),(1,4),(2,5),(3,6),(4,7),(5,8),(6,9),
    (7,10),(8,11),(9,12),(12,13),(12,14),(12,15),(13,16),(14,17),
    (16,18),(17,19),(18,20),(19,21),
]
# Only trail these joints (clean, not smudgy)
TRAIL_JOINTS = {7, 8, 10, 11, 15, 20, 21}  # ankles, feet, head, wrists

BASE = Path(__file__).parent
MARGIN = 40
TL_W = W - 2 * MARGIN


# ── Data loading ──────────────────────────────────────────────────

def load_all(joints_path=None, metrics_path=None):
    r = BASE / "results"
    joints = np.load(joints_path or (r / "joints_3d_CLEAN.npy"))

    with open(metrics_path or (r / "CLEAN_metrics.json")) as f:
        metrics = json.load(f)

    beats = np.array([])
    beats_file = BASE.parent / "data/brace/annotations/audio_beats.json"
    if beats_file.exists():
        with open(beats_file) as f:
            beats = np.array(json.load(f)["RS0mFARO1x4.4"]["beats_sec"])

    segments = []
    seg_file = BASE.parent / "data/brace/annotations/segments.csv"
    if seg_file.exists():
        with open(seg_file) as f:
            for row in csv.DictReader(f):
                if row["video_id"] == "RS0mFARO1x4" and int(row["seq_idx"]) == 4:
                    segments.append({
                        "start_s": (int(row["start_frame"]) - 3802) / FPS,
                        "end_s": (int(row["end_frame"]) - 3802) / FPS,
                        "dance_type": row["dance_type"],
                    })

    # Audio energy
    perc = np.load(r / "audio_percussive.npy")
    return joints, metrics, beats, segments, perc


def compute_beat_alignment(joints, beats, fps, window_ms=100):
    dt = 1.0 / fps
    vel = np.diff(joints, axis=0) / dt
    speed = np.linalg.norm(vel, axis=-1).sum(axis=1)

    from scipy.signal import find_peaks, savgol_filter
    w = min(21, len(speed) // 2 * 2 - 1)
    smooth = savgol_filter(speed, w, 3) if w >= 5 else speed
    peaks, _ = find_peaks(smooth, distance=int(fps * 0.2))
    peak_times = peaks / fps

    total_dur = len(joints) / fps
    hits = []
    for bt in beats:
        if 0 <= bt < total_dur:
            aligned = any(abs(pt - bt) < window_ms / 1000 for pt in peak_times)
            hits.append({"time": float(bt), "aligned": aligned})
    return hits


# ── Three-view renderer ──────────────────────────────────────────

class TriViewRenderer:
    """Draws front/side/top-down skeleton + trails in 3 panels."""

    def __init__(self, joints, segments):
        self.joints = joints
        self.segments = segments
        self.n_frames = joints.shape[0]
        self.n_joints = joints.shape[1]

        # Detect identity swaps
        pelvis = joints[:, 0, :]
        disp = np.linalg.norm(np.diff(pelvis, axis=0), axis=-1)
        self.swap_frames = set(np.where(disp > SWAP_THRESH)[0].tolist())

        # Global bounding boxes for each view (stable projection)
        x, y, z = joints[:,:,0], joints[:,:,1], joints[:,:,2]
        pad = 0.4
        self.bounds = {
            "front": (x.min()-pad, x.max()+pad, y.min()-pad, y.max()+pad),  # XY
            "side":  (z.min()-pad, z.max()+pad, y.min()-pad, y.max()+pad),  # ZY
            "top":   (x.min()-pad, x.max()+pad, z.min()-pad, z.max()+pad),  # XZ
        }

        # COM trail
        self.com = joints[:, 0, :]  # pelvis as COM proxy

        self.f_label = ImageFont.truetype(FONT_B, 14)
        self.f_scale = ImageFont.truetype(FONT_R, 11)

    def _project(self, view, point, panel_x, panel_y, pw, ph):
        """Project 3D point to 2D in the given view."""
        if view == "front":
            u, v = point[0], point[1]
        elif view == "side":
            u, v = point[2], point[1]
        else:  # top
            u, v = point[0], point[2]

        b = self.bounds[view]
        ufrac = (u - b[0]) / (b[1] - b[0])
        vfrac = (v - b[2]) / (b[3] - b[2])

        px = panel_x + int(ufrac * pw)
        if view == "top":
            py = panel_y + int(vfrac * ph)  # Z increases downward in top view
        else:
            py = panel_y + ph - int(vfrac * ph)  # Y-up for front/side
        return px, py

    def _has_swap_between(self, fa, fb):
        for f in range(min(fa, fb), max(fa, fb)):
            if f in self.swap_frames:
                return True
        return False

    def draw_view(self, od, view, panel_x, panel_y, pw, ph, frame_idx):
        """Draw one orthographic view panel."""
        # Panel background
        od.rectangle([panel_x, panel_y, panel_x + pw, panel_y + ph], fill=(12, 12, 18))

        # Grid lines (1m spacing)
        b = self.bounds[view]
        for val in np.arange(np.ceil(b[0]), b[1], 1.0):
            frac = (val - b[0]) / (b[1] - b[0])
            gx = panel_x + int(frac * pw)
            od.line([(gx, panel_y), (gx, panel_y + ph)], fill=(*GRID_C, 80), width=1)
        for val in np.arange(np.ceil(b[2]), b[3], 1.0):
            frac = (val - b[2]) / (b[3] - b[2])
            if view == "top":
                gy = panel_y + int(frac * ph)
            else:
                gy = panel_y + ph - int(frac * ph)
            od.line([(panel_x, gy), (panel_x + pw, gy)], fill=(*GRID_C, 80), width=1)

        # View label
        labels = {"front": "FRONT (XY)", "side": "SIDE (ZY)", "top": "TOP-DOWN (XZ)"}
        od.text((panel_x + 8, panel_y + 6), labels[view], fill=TXT_DIM, font=self.f_label)

        # ── Trails (endpoint joints only, no smudge) ──
        trail_start = max(0, frame_idx - TRAIL_LEN)
        trail_frames = frame_idx - trail_start

        if trail_frames > 1:
            for t_idx in range(trail_start, frame_idx):
                if self._has_swap_between(t_idx, frame_idx):
                    continue
                progress = (t_idx - trail_start) / max(1, trail_frames)
                alpha = int(20 + 180 * progress)
                dot_r = max(1, int(1 + 3 * progress))

                j = self.joints[t_idx]
                for ji in TRAIL_JOINTS:
                    if ji >= self.n_joints:
                        continue
                    color = CLUSTER_COLORS[JOINT_CLUSTER.get(ji, "torso")]
                    px, py = self._project(view, j[ji], panel_x, panel_y, pw, ph)
                    od.ellipse([px-dot_r, py-dot_r, px+dot_r, py+dot_r],
                              fill=(*color, min(alpha, 255)))

        # ── COM trail (3s, red) ──
        com_trail = min(90, frame_idx)
        com_start = max(0, frame_idx - com_trail)
        for ci in range(com_start, frame_idx):
            if self._has_swap_between(ci, frame_idx):
                continue
            prog = (ci - com_start) / max(1, com_trail)
            alpha = int(30 + 180 * prog)
            px, py = self._project(view, self.com[ci], panel_x, panel_y, pw, ph)
            r = max(1, int(2 * prog))
            od.ellipse([px-r, py-r, px+r, py+r], fill=(*C_COM, min(alpha, 255)))

        # ── Current skeleton ──
        j_now = self.joints[frame_idx]
        for j1, j2 in BONES:
            if j1 >= self.n_joints or j2 >= self.n_joints:
                continue
            px1, py1 = self._project(view, j_now[j1], panel_x, panel_y, pw, ph)
            px2, py2 = self._project(view, j_now[j2], panel_x, panel_y, pw, ph)
            c = CLUSTER_COLORS[JOINT_CLUSTER.get(j1, "torso")]
            od.line([(px1,py1),(px2,py2)], fill=(*c, 200), width=2)

        # Current joint dots
        for ji in range(self.n_joints):
            color = CLUSTER_COLORS[JOINT_CLUSTER.get(ji, "torso")]
            px, py = self._project(view, j_now[ji], panel_x, panel_y, pw, ph)
            r = 5 if ji in TRAIL_JOINTS else 3
            od.ellipse([px-r, py-r, px+r, py+r], fill=(*color, 255))

        # COM current (bright red, large)
        cpx, cpy = self._project(view, self.com[frame_idx], panel_x, panel_y, pw, ph)
        od.ellipse([cpx-6, cpy-6, cpx+6, cpy+6], fill=(*C_COM, 255), outline=(255,255,255,180))

        # Panel border
        od.rectangle([panel_x, panel_y, panel_x + pw, panel_y + ph], outline=(50, 50, 60))


# ── Bottom rows ───────────────────────────────────────────────────

def draw_move_energy_row(d, y, segments, energy, progress, total_dur, beats, font_b, font_sm):
    """Move type bar with energy fill behind it."""
    d.rectangle([0, y, W, y + MOVE_ROW_H], fill=(16, 16, 22))

    # Energy curve as filled area
    e_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
    x_idx = np.linspace(0, len(e_norm) - 1, TL_W).astype(int)
    for x in range(TL_W):
        val = float(e_norm[x_idx[x]])
        bar_h = max(1, int(val * (MOVE_ROW_H - 20)))
        px = MARGIN + x
        frac = x / TL_W
        dim = 1.0 if frac <= progress else 0.25

        # Color from move type at this position
        t_at = frac * total_dur
        seg_color = (80, 80, 80)
        for seg in segments:
            if seg["start_s"] <= t_at <= seg["end_s"]:
                seg_color = SEG_COLORS.get(seg["dance_type"], (80, 80, 80))
                break

        r = int(seg_color[0] * dim)
        g = int(seg_color[1] * dim)
        b = int(seg_color[2] * dim)
        base = y + MOVE_ROW_H - 5
        d.line([(px, base - bar_h), (px, base)], fill=(r, g, b))

    # Move type labels on top
    for seg in segments:
        xs = MARGIN + int(seg["start_s"] / total_dur * TL_W)
        xe = MARGIN + int(seg["end_s"] / total_dur * TL_W)
        label = seg["dance_type"].upper()
        if xe - xs > 50:
            d.text((xs + 4, y + 4), label, fill=TXT, font=font_sm)

    # Beat markers (subtle vertical lines)
    for bt in beats:
        if 0 <= bt < total_dur:
            bx = MARGIN + int(bt / total_dur * TL_W)
            d.line([(bx, y + MOVE_ROW_H - 8), (bx, y + MOVE_ROW_H - 2)], fill=(80, 80, 90), width=1)

    # Playhead
    ph = MARGIN + int(progress * TL_W)
    d.line([(ph, y), (ph, y + MOVE_ROW_H)], fill=TXT, width=2)

    # Label
    d.text((4, y + 4), "ENERGY", fill=TXT_DIM, font=font_sm)


def draw_musicality_row(d, y, beat_hits, progress, total_dur, mu_val, font_big, font_med, font_sm):
    """Musicality hero row: big stat + beat dot strip."""
    d.rectangle([0, y, W, y + MU_ROW_H], fill=(16, 16, 22))

    hits_so_far = sum(1 for bh in beat_hits if bh["aligned"] and bh["time"] <= progress * total_dur)
    total_so_far = sum(1 for bh in beat_hits if bh["time"] <= progress * total_dur)
    total_beats = len(beat_hits)
    total_hits = sum(1 for bh in beat_hits if bh["aligned"])
    pct = (total_hits / max(total_beats, 1)) * 100

    # Big stat left
    stat_text = f"{hits_so_far}/{total_so_far}"
    d.text((MARGIN, y + 8), stat_text, fill=TXT, font=font_big)
    stat_w = font_big.getlength(stat_text)
    d.text((MARGIN + stat_w + 8, y + 18), "BEATS HIT", fill=TXT_DIM, font=font_sm)

    # Percentage
    pct_now = (hits_so_far / max(total_so_far, 1)) * 100 if total_so_far > 0 else 0
    pct_text = f"({pct_now:.0f}%)"
    d.text((MARGIN + stat_w + 90, y + 18), pct_text, fill=C_BEAT_HIT, font=font_sm)

    # Beat dot strip (center)
    dot_x_start = 380
    dot_spacing = max(6, min(14, (W - 600) // max(total_beats, 1)))
    dot_y = y + 50

    for i, bh in enumerate(beat_hits):
        dx = dot_x_start + i * dot_spacing
        if dx > W - 200:
            break
        if bh["time"] <= progress * total_dur:
            color = C_BEAT_HIT if bh["aligned"] else C_BEAT_MISS
        else:
            color = (35, 35, 42)
        d.ellipse([dx - 3, dot_y - 3, dx + 3, dot_y + 3], fill=color)

    # Mu value right
    mu_text = f"\u03bc = {mu_val:.2f}"
    d.text((W - MARGIN - 120, y + 10), mu_text, fill=C_MU, font=font_big)

    # Playhead line
    ph = MARGIN + int(progress * TL_W)
    d.line([(ph, y), (ph, y + MU_ROW_H)], fill=TXT, width=2)


def draw_footer(d, y, bpm, total_dur, peak_vel, total_beats, total_hits, font_sm):
    """Stats footer with real units."""
    d.rectangle([0, y, W, y + FOOTER_H], fill=(10, 10, 14))
    hit_pct = total_hits / max(total_beats, 1) * 100

    stats = (f"{bpm:.0f} BPM  |  Round: {total_dur:.1f}s  |  "
             f"Peak: {peak_vel:.1f} m/s  |  "
             f"{total_beats} beats  |  {total_hits}/{total_beats} hit ({hit_pct:.0f}%)")
    d.text((MARGIN, y + 12), stats, fill=TXT_DIM, font=font_sm)


# ── Main render ───────────────────────────────────────────────────

def render(joints_path=None, metrics_path=None, mesh_video_path=None, audio_path=None, output_path=None):
    output = output_path or str(BASE / "exports/v2/trails.mp4")
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    joints, metrics, beats, segments, perc = load_all(joints_path, metrics_path)
    n_frames = joints.shape[0]
    total_dur = n_frames / FPS

    # Compute stats
    if len(beats) > 1:
        bpm = 60.0 / np.median(np.diff(beats))
    else:
        bpm = 0

    dt = 1.0 / FPS
    vel = np.diff(joints, axis=0) / dt
    speed = np.linalg.norm(vel, axis=-1)
    energy = speed.sum(axis=1)
    from scipy.signal import savgol_filter
    w = min(31, len(energy) // 2 * 2 - 1)
    energy_smooth = savgol_filter(energy, w, 3) if w >= 5 else energy
    # Peak COM velocity (pelvis), not max joint velocity
    com_vel = np.linalg.norm(np.diff(joints[:, 0, :], axis=0) / dt, axis=-1)
    peak_vel = float(com_vel.max())

    print("Computing beat alignment...")
    beat_hits = compute_beat_alignment(joints, beats, FPS)
    total_beats = len(beat_hits)
    total_hits = sum(1 for bh in beat_hits if bh["aligned"])

    # Compute mu on clean frames only (first 500 frames, before identity swaps)
    sys.path.insert(0, str(BASE.parent / "src"))
    try:
        from extreme_motion_reimpl.recap.metrics import compute_musicality
        clean_end = 500  # Before identity swaps start at frame 504
        clean_joints = joints[:clean_end]
        clean_beats = beats[(beats >= 0) & (beats < clean_end / FPS)]
        if len(clean_beats) >= 2:
            m = compute_musicality(clean_joints, clean_beats, FPS, sg_window=11)
            mu_val = m["mu"]
        else:
            mu_val = 0.0
    except:
        mu_val = 0.0
    print(f"  Clean-frames \u03bc = {mu_val:.3f} (frames 0-{clean_end})")

    renderer = TriViewRenderer(joints, segments)

    # Fonts
    f_big = ImageFont.truetype(FONT_B, 28)
    f_med = ImageFont.truetype(FONT_B, 18)
    f_sm = ImageFont.truetype(FONT_R, 14)
    f_stats = ImageFont.truetype(FONT_B, 16)

    # Source video
    mesh_video = mesh_video_path or str(BASE / "results/gvhmr_mesh_clean_seq4.mp4")
    probe = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                           "-show_streams", mesh_video], capture_output=True, text=True)
    vs = [s for s in json.loads(probe.stdout)["streams"] if s["codec_type"] == "video"][0]
    src_w, src_h = int(vs["width"]), int(vs["height"])
    src_bytes = src_w * src_h * 3

    audio_tmp = tempfile.NamedTemporaryFile(suffix=".aac", delete=False)
    audio_src = audio_path or str(BASE.parent / "josh_input/bcone_seq4/audio.wav")
    subprocess.run(["ffmpeg", "-y", "-i", audio_src, "-vn", "-c:a", "aac",
                    "-b:a", "192k", audio_tmp.name], capture_output=True)

    read_proc = subprocess.Popen(
        ["ffmpeg", "-i", mesh_video, "-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "error", "pipe:1"],
        stdout=subprocess.PIPE, bufsize=src_bytes * 2)

    write_proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "pipe:0", "-i", audio_tmp.name,
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "aac",
         "-pix_fmt", "yuv420p", "-shortest", "-v", "error", output],
        stdin=subprocess.PIPE, bufsize=W * H * 3 * 2)

    print(f"Rendering {n_frames} frames @ {W}x{H} — v3 'The Breakdown'...")
    t0 = time.time()

    for fi in range(n_frames + 10):
        raw = read_proc.stdout.read(src_bytes)
        if len(raw) < src_bytes:
            break
        if fi >= n_frames:
            continue

        progress = fi / n_frames
        t = fi / FPS
        canvas = Image.new("RGB", (W, H), BG)

        # ── Top: Video ────────────────────────────────────────────
        vf = Image.frombytes("RGB", (src_w, src_h), raw).resize((W, VID_H), Image.LANCZOS)
        canvas.paste(vf, (0, 0))

        # Stats badge (top-left on video)
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle([15, 12, 170, 80], radius=8, fill=(0, 0, 0, 200))
        od.text((25, 16), f"{bpm:.0f} BPM", fill=TXT, font=f_stats)
        od.text((25, 36), f"{t:.1f}s / {total_dur:.1f}s", fill=TXT_DIM, font=f_sm)
        od.text((25, 54), f"{total_beats} beats", fill=TXT_DIM, font=f_sm)

        # Move type label (bottom-left on video)
        for seg in segments:
            if seg["start_s"] <= t <= seg["end_s"]:
                c = SEG_COLORS.get(seg["dance_type"], (128, 128, 128))
                label = seg["dance_type"].upper()
                tw = f_med.getlength(label)
                od.rounded_rectangle([15, VID_H - 42, 15 + tw + 20, VID_H - 12],
                                    radius=6, fill=(*c, 200))
                od.text((25, VID_H - 38), label, fill=TXT, font=f_med)
                break

        # ── Mid: Three views ──────────────────────────────────────
        view_y = VID_H
        renderer.draw_view(od, "front", 0, view_y, VIEW_W, VIEW_H, fi)
        renderer.draw_view(od, "side", VIEW_W, view_y, VIEW_W, VIEW_H, fi)
        renderer.draw_view(od, "top", VIEW_W * 2, view_y, VIEW_W, VIEW_H, fi)

        # Composite RGBA overlay
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

        # ── Bottom rows (regular draw, no alpha needed) ───────────
        d = ImageDraw.Draw(canvas)
        row_y = VID_H + VIEW_H

        draw_move_energy_row(d, row_y, segments, energy_smooth, progress, total_dur, beats, f_med, f_sm)

        row_y += MOVE_ROW_H
        draw_musicality_row(d, row_y, beat_hits, progress, total_dur, mu_val, f_big, f_med, f_sm)

        row_y += MU_ROW_H
        draw_footer(d, row_y, bpm, total_dur, peak_vel, total_beats, total_hits, f_sm)

        write_proc.stdin.write(canvas.tobytes())

        if fi % 100 == 0 and fi > 0:
            elapsed = time.time() - t0
            print(f"  {fi}/{n_frames} ({100*fi/n_frames:.0f}%) \u2014 {fi/elapsed:.1f} fps")

    write_proc.stdin.close()
    write_proc.wait()
    read_proc.wait()
    elapsed = time.time() - t0
    print(f"\nDone! {output} ({elapsed:.0f}s)")

    # Provenance: save timestamped copy
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned = Path(output).parent / f"trails_v3_{ts}.mp4"
    shutil.copy2(output, versioned)
    print(f"Versioned: {versioned}")

    Path(audio_tmp.name).unlink(missing_ok=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ghost Trails v3 — The Breakdown")
    parser.add_argument("--joints", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--mesh-video", default=None)
    parser.add_argument("--audio", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    render(joints_path=args.joints, metrics_path=args.metrics,
           mesh_video_path=args.mesh_video, audio_path=args.audio,
           output_path=args.output)
