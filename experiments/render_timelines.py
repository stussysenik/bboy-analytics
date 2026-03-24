"""
Timelines Renderer (v2) — Audio stems + Energy + Flow + Running μ + Beat dots.

Layout (1920x1080):
  Top 432px:    Battle video (scaled)
  Row 1 (80px): Percussive waveform (kicks/drums) — orange
  Row 2 (80px): Harmonic waveform (vocals/melody) — blue
  Row 3 (80px): Energy curve (gradient blue→red)
  Row 4 (80px): Flow smoothness + Running μ curve + beat alignment dots
  Row 5 (30px): Segment color bar + timer
  Bottom 18px:  Legend

Output: experiments/exports/timelines/
"""

from __future__ import annotations
import json, subprocess, sys, time, csv, tempfile
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 29.97
VID_H = 432
ROW_H = 80
SEG_H = 48
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

BG = (14, 14, 20)
GRID = (40, 40, 50)
TXT = (255, 255, 255)
TXT_DIM = (140, 140, 155)
C_PERC = (255, 140, 50)   # Orange for percussive
C_HARM = (80, 160, 255)   # Blue for harmonic
C_ENERGY_LO = (50, 130, 200)
C_ENERGY_HI = (255, 80, 40)
C_MU = (180, 100, 255)    # Purple for μ
C_BEAT_HIT = (50, 220, 100)  # Green
C_BEAT_MISS = (220, 60, 60)  # Red
C_PLAYHEAD = (255, 255, 255)
SEG_COLORS = {"toprock": (33, 150, 243), "footwork": (76, 175, 80), "powermove": (255, 87, 34)}

BASE = Path(__file__).parent
MARGIN = 40  # Left/right margin for timelines
TL_W = W - 2 * MARGIN  # Timeline width


def load_all(joints_path=None, metrics_path=None):
    r = BASE / "results"
    joints = np.load(joints_path or (r / "joints_3d_CLEAN.npy"))
    perc = np.load(r / "audio_percussive.npy")
    harm = np.load(r / "audio_harmonic.npy")

    with open(metrics_path or (r / "CLEAN_metrics.json")) as f:
        metrics = json.load(f)
    with open(BASE.parent / "data/brace/annotations/audio_beats.json") as f:
        beats = np.array(json.load(f)["RS0mFARO1x4.4"]["beats_sec"])

    segments = []
    with open(BASE.parent / "data/brace/annotations/segments.csv") as f:
        for row in csv.DictReader(f):
            if row["video_id"] == "RS0mFARO1x4" and int(row["seq_idx"]) == 4:
                segments.append({
                    "start_s": (int(row["start_frame"]) - 3802) / FPS,
                    "end_s": (int(row["end_frame"]) - 3802) / FPS,
                    "dance_type": row["dance_type"],
                })
    return joints, perc, harm, metrics, beats, segments


def compute_running_mu(joints, beats, fps, window_s=4.0):
    """Compute μ in a sliding window across the clip."""
    sys.path.insert(0, str(BASE.parent / "src"))
    from extreme_motion_reimpl.recap.metrics import compute_musicality
    n = joints.shape[0]
    window = int(window_s * fps)
    mus = np.zeros(n)
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2)
        if end - start < 30:
            continue
        seg = joints[start:end]
        local_beats = beats[(beats >= start / fps) & (beats < end / fps)] - start / fps
        if len(local_beats) < 2:
            continue
        try:
            m = compute_musicality(seg, local_beats, fps, sg_window=11)
            mus[i] = m["mu"]
        except:
            pass
    return mus


def compute_beat_alignment(joints, beats, fps, window_ms=100):
    """For each beat, check if there's a velocity peak within window_ms."""
    dt = 1.0 / fps
    vel = np.diff(joints, axis=0) / dt
    speed = np.linalg.norm(vel, axis=-1).sum(axis=1)  # total energy per frame

    # Find peaks in energy
    from scipy.signal import find_peaks, savgol_filter
    w = min(21, len(speed) // 2 * 2 - 1)
    if w >= 5:
        smooth = savgol_filter(speed, w, 3)
    else:
        smooth = speed
    peaks, _ = find_peaks(smooth, distance=int(fps * 0.2))
    peak_times = peaks / fps

    hits = []
    for bt in beats:
        if 0 <= bt < len(speed) / fps:
            aligned = any(abs(pt - bt) < window_ms / 1000 for pt in peak_times)
            hits.append({"time": float(bt), "aligned": aligned})
    return hits


def draw_waveform_row(d, y, data, color, label, progress, beats, total_dur):
    """Draw one waveform timeline row."""
    # Background
    d.rectangle([0, y, W, y + ROW_H], fill=(20, 20, 28))
    d.line([(0, y), (W, y)], fill=GRID, width=1)

    # Label
    font_sm = ImageFont.truetype(FONT_B, 14)
    d.text((8, y + 4), label, fill=TXT_DIM, font=font_sm)

    # Waveform
    norm = data / (data.max() + 1e-8)
    x_idx = np.linspace(0, len(norm) - 1, TL_W).astype(int)
    mid = y + ROW_H // 2
    for x in range(TL_W):
        val = float(norm[x_idx[x]])
        bar_h = max(1, int(val * ROW_H * 0.7))
        px = MARGIN + x
        # Dim future, bright past
        frac = x / TL_W
        if frac <= progress:
            alpha_mult = 1.0
        else:
            alpha_mult = 0.3
        r = int(color[0] * alpha_mult)
        g = int(color[1] * alpha_mult)
        b = int(color[2] * alpha_mult)
        d.line([(px, mid - bar_h // 2), (px, mid + bar_h // 2)], fill=(r, g, b))

    # Beat markers
    for bt in beats:
        if 0 <= bt < total_dur:
            bx = MARGIN + int(bt / total_dur * TL_W)
            d.line([(bx, y + 2), (bx, y + ROW_H - 2)], fill=(*C_PLAYHEAD, 60)[:3], width=1)

    # Playhead
    ph = MARGIN + int(progress * TL_W)
    d.line([(ph, y), (ph, y + ROW_H)], fill=C_PLAYHEAD, width=2)


def draw_energy_row(d, y, energy, progress, total_dur):
    """Draw energy curve timeline."""
    d.rectangle([0, y, W, y + ROW_H], fill=(20, 20, 28))
    d.line([(0, y), (W, y)], fill=GRID, width=1)

    font_sm = ImageFont.truetype(FONT_B, 14)
    d.text((8, y + 4), "ENERGY", fill=TXT_DIM, font=font_sm)

    e_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
    x_idx = np.linspace(0, len(e_norm) - 1, TL_W).astype(int)

    for x in range(TL_W):
        val = float(e_norm[x_idx[x]])
        bar_h = max(1, int(val * ROW_H * 0.7))
        px = MARGIN + x
        frac = x / TL_W
        dim = 1.0 if frac <= progress else 0.3
        r = int((C_ENERGY_LO[0] + (C_ENERGY_HI[0] - C_ENERGY_LO[0]) * val) * dim)
        g = int((C_ENERGY_LO[1] + (C_ENERGY_HI[1] - C_ENERGY_LO[1]) * val) * dim)
        b = int((C_ENERGY_LO[2] + (C_ENERGY_HI[2] - C_ENERGY_LO[2]) * val) * dim)
        base_y = y + ROW_H - 10
        d.line([(px, base_y - bar_h), (px, base_y)], fill=(r, g, b))

    ph = MARGIN + int(progress * TL_W)
    d.line([(ph, y), (ph, y + ROW_H)], fill=C_PLAYHEAD, width=2)


def draw_mu_row(d, y, running_mu, beat_hits, progress, total_dur):
    """Draw running μ curve + beat alignment dots."""
    d.rectangle([0, y, W, y + ROW_H], fill=(20, 20, 28))
    d.line([(0, y), (W, y)], fill=GRID, width=1)

    font_sm = ImageFont.truetype(FONT_B, 14)
    d.text((8, y + 4), "MUSICALITY (μ)", fill=TXT_DIM, font=font_sm)

    # Running μ curve
    n_show = int(progress * len(running_mu))
    if n_show > 1:
        x_pts = np.linspace(0, TL_W, n_show).astype(int)
        mu_max = max(running_mu.max(), 0.1)
        prev_px, prev_py = None, None
        for i in range(n_show):
            px = MARGIN + x_pts[i]
            val = running_mu[i] / mu_max
            py = int(y + ROW_H - 15 - val * (ROW_H - 25))
            if prev_px is not None:
                d.line([(prev_px, prev_py), (px, py)], fill=C_MU, width=2)
            prev_px, prev_py = px, py

    # 0.3 threshold line
    if running_mu.max() > 0:
        thresh_y = int(y + ROW_H - 15 - (0.3 / max(running_mu.max(), 0.1)) * (ROW_H - 25))
        if y < thresh_y < y + ROW_H:
            d.line([(MARGIN, thresh_y), (MARGIN + TL_W, thresh_y)],
                   fill=(100, 100, 100), width=1)
            d.text((MARGIN + TL_W + 5, thresh_y - 7), "0.3", fill=TXT_DIM, font=font_sm)

    # Beat alignment dots
    for bh in beat_hits:
        if bh["time"] <= progress * total_dur:
            bx = MARGIN + int(bh["time"] / total_dur * TL_W)
            color = C_BEAT_HIT if bh["aligned"] else C_BEAT_MISS
            d.ellipse([bx - 4, y + ROW_H - 12, bx + 4, y + ROW_H - 4], fill=color)

    # Running count
    hits_so_far = sum(1 for bh in beat_hits if bh["aligned"] and bh["time"] <= progress * total_dur)
    total_so_far = sum(1 for bh in beat_hits if bh["time"] <= progress * total_dur)
    if total_so_far > 0:
        pct = hits_so_far / total_so_far * 100
        font_med = ImageFont.truetype(FONT_B, 16)
        d.text((W - 200, y + 4), f"{hits_so_far}/{total_so_far} aligned ({pct:.0f}%)",
               fill=TXT, font=font_med)

    ph = MARGIN + int(progress * TL_W)
    d.line([(ph, y), (ph, y + ROW_H)], fill=C_PLAYHEAD, width=2)


def draw_segment_bar(d, y, segments, progress, total_dur):
    """Draw segment color bar + timer."""
    d.rectangle([0, y, W, y + SEG_H], fill=(14, 14, 20))

    font_med = ImageFont.truetype(FONT_B, 16)
    font_sm = ImageFont.truetype(FONT_R, 12)

    for seg in segments:
        xs = MARGIN + int(seg["start_s"] / total_dur * TL_W)
        xe = MARGIN + int(seg["end_s"] / total_dur * TL_W)
        c = SEG_COLORS.get(seg["dance_type"], (128, 128, 128))
        d.rectangle([xs, y + 4, xe, y + 24], fill=c)
        # Label inside
        label = seg["dance_type"].upper()[:8]
        if xe - xs > 60:
            d.text((xs + 4, y + 6), label, fill=TXT, font=font_sm)

    # Timer
    t = progress * total_dur
    d.text((W - 160, y + 8), f"{t:.1f}s / {total_dur:.1f}s",
           fill=TXT, font=font_med)

    # Legend
    ly = y + 30
    items = [("PERC", C_PERC), ("HARM", C_HARM), ("ENERGY", C_ENERGY_HI),
             ("μ", C_MU), ("HIT", C_BEAT_HIT), ("MISS", C_BEAT_MISS)]
    lx = MARGIN
    for label, color in items:
        d.rectangle([lx, ly, lx + 10, ly + 10], fill=color)
        d.text((lx + 14, ly - 1), label, fill=TXT_DIM, font=font_sm)
        lx += font_sm.getlength(label) + 28


def render(joints_path=None, metrics_path=None, mesh_video_path=None, audio_path=None, output_path=None):
    output = output_path or str(BASE / "exports/timelines/gvhmr.mp4")
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    joints, perc, harm, metrics, beats, segments = load_all(joints_path, metrics_path)
    n_frames = joints.shape[0]
    total_dur = n_frames / FPS

    # Compute energy from joints
    dt = 1.0 / FPS
    vel = np.diff(joints, axis=0) / dt
    speed = np.linalg.norm(vel, axis=-1)
    energy = speed.sum(axis=1)
    from scipy.signal import savgol_filter
    w = min(31, len(energy) // 2 * 2 - 1)
    energy_smooth = savgol_filter(energy, w, 3) if w >= 5 else energy

    print("Computing running μ (sliding window)...")
    running_mu = compute_running_mu(joints, beats, FPS, window_s=4.0)

    print("Computing beat alignment...")
    beat_hits = compute_beat_alignment(joints, beats, FPS)

    # Source video
    mesh_video = mesh_video_path or str(BASE / "results/gvhmr_mesh_clean_seq4.mp4")
    probe = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                           "-show_streams", mesh_video], capture_output=True, text=True)
    vs = [s for s in json.loads(probe.stdout)["streams"] if s["codec_type"] == "video"][0]
    src_w, src_h = int(vs["width"]), int(vs["height"])
    src_bytes = src_w * src_h * 3

    # Audio
    audio_tmp = tempfile.NamedTemporaryFile(suffix=".aac", delete=False)
    audio_src = audio_path or "/tmp/lilg_round.mp4"
    subprocess.run(["ffmpeg", "-y", "-i", audio_src, "-vn", "-c:a", "aac",
                    "-b:a", "192k", audio_tmp.name], capture_output=True)

    # FFmpeg pipes
    read_proc = subprocess.Popen(
        ["ffmpeg", "-i", mesh_video, "-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "error", "pipe:1"],
        stdout=subprocess.PIPE, bufsize=src_bytes * 2)

    write_proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "pipe:0", "-i", audio_tmp.name,
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "aac",
         "-pix_fmt", "yuv420p", "-shortest", "-v", "error", output],
        stdin=subprocess.PIPE, bufsize=W * H * 3 * 2)

    print(f"Rendering {n_frames} frames...")
    t0 = time.time()

    for fi in range(n_frames + 10):  # slight overread tolerance
        raw = read_proc.stdout.read(src_bytes)
        if len(raw) < src_bytes:
            break
        if fi >= n_frames:
            continue

        progress = fi / n_frames
        canvas = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(canvas)

        # Top: video
        vf = Image.frombytes("RGB", (src_w, src_h), raw).resize((W, VID_H), Image.LANCZOS)
        canvas.paste(vf, (0, 0))

        # Timeline rows
        row_y = VID_H
        draw_waveform_row(d, row_y, perc, C_PERC, "PERCUSSIVE (kicks/drums)",
                         progress, beats, total_dur)

        row_y += ROW_H
        draw_waveform_row(d, row_y, harm, C_HARM, "HARMONIC (vocals/melody)",
                         progress, beats, total_dur)

        row_y += ROW_H
        draw_energy_row(d, row_y, energy_smooth, progress, total_dur)

        row_y += ROW_H
        draw_mu_row(d, row_y, running_mu, beat_hits, progress, total_dur)

        row_y += ROW_H
        draw_segment_bar(d, row_y, segments, progress, total_dur)

        write_proc.stdin.write(canvas.tobytes())

        if fi % 100 == 0 and fi > 0:
            elapsed = time.time() - t0
            fps_actual = fi / elapsed
            print(f"  {fi}/{n_frames} ({100*fi/n_frames:.0f}%) — {fps_actual:.1f} fps")

    write_proc.stdin.close()
    write_proc.wait()
    read_proc.wait()

    elapsed = time.time() - t0
    print(f"\nDone! {output} ({elapsed:.0f}s)")
    Path(audio_tmp.name).unlink(missing_ok=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--joints", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--mesh-video", default=None)
    parser.add_argument("--audio", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    render(joints_path=args.joints, metrics_path=args.metrics,
           mesh_video_path=args.mesh_video, audio_path=args.audio,
           output_path=args.output)
