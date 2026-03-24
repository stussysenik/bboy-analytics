"""
RendererBase: shared ffmpeg pipeline, frame loop, layout composition.

All renderers use this to avoid duplicating pipe setup, audio muxing,
and progress reporting.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from PIL import Image

from .panel import Panel, BG


class RendererBase:
    """Composites panels into a final video via ffmpeg pipes."""

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
    ):
        self.W = width
        self.H = height
        self.fps = fps
        self.panels: list[tuple[Panel, int, int]] = []
        self.video_overlay = None  # special: draws ON the video frame
        self.video_rect = (0, 0, width, 640)  # x, y, w, h for video area

    def add_panel(self, panel: Panel, x: int, y: int) -> None:
        """Add a panel at absolute position (x, y) in the output frame."""
        self.panels.append((panel, x, y))

    def set_video_overlay(self, overlay_panel) -> None:
        """Set a panel that draws directly on the video frame (overlay)."""
        self.video_overlay = overlay_panel

    def set_video_rect(self, x: int, y: int, w: int, h: int) -> None:
        """Define where the source video is placed."""
        self.video_rect = (x, y, w, h)

    def prerender_all(self) -> None:
        """Call prerender on all panels."""
        for panel, _, _ in self.panels:
            panel.prerender()
        if self.video_overlay:
            self.video_overlay.prerender()

    def render(
        self,
        source_video: str,
        output_path: str,
        n_frames: int,
        audio_source: str | None = None,
    ) -> str:
        """
        Render the full video.

        Args:
            source_video: path to source mesh/overlay video
            output_path: output MP4 path
            n_frames: number of frames to render
            audio_source: optional path to video/audio for audio track
        """
        # Probe source video
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", source_video],
            capture_output=True, text=True)
        streams = json.loads(probe.stdout)["streams"]
        vs = [s for s in streams if s["codec_type"] == "video"][0]
        src_w, src_h = int(vs["width"]), int(vs["height"])
        src_bytes = src_w * src_h * 3

        # Extract audio if provided
        audio_tmp = None
        audio_args = []
        if audio_source and Path(audio_source).exists():
            audio_tmp = tempfile.NamedTemporaryFile(suffix=".aac", delete=False)
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_source, "-vn", "-c:a", "aac",
                 "-b:a", "192k", audio_tmp.name],
                capture_output=True)
            audio_args = ["-i", audio_tmp.name, "-c:a", "aac", "-b:a", "192k", "-shortest"]

        # FFmpeg pipes
        read_proc = subprocess.Popen(
            ["ffmpeg", "-i", source_video, "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-v", "error", "pipe:1"],
            stdout=subprocess.PIPE, bufsize=src_bytes * 2)

        out_bytes = self.W * self.H * 3
        write_cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.W}x{self.H}", "-r", str(self.fps),
            "-i", "pipe:0", *audio_args,
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-v", "error", output_path,
        ]
        write_proc = subprocess.Popen(
            write_cmd, stdin=subprocess.PIPE, bufsize=out_bytes * 2)

        # Prerender static elements
        self.prerender_all()

        # Frame loop
        print(f"Rendering {n_frames} frames → {output_path}")
        t0 = time.time()
        frame_idx = 0

        while frame_idx < n_frames:
            raw = read_proc.stdout.read(src_bytes)
            if len(raw) < src_bytes:
                break

            video_frame = Image.frombytes("RGB", (src_w, src_h), raw)
            canvas = self._compose_frame(frame_idx, video_frame)
            write_proc.stdin.write(canvas.tobytes())

            frame_idx += 1
            if frame_idx % 150 == 0:
                elapsed = time.time() - t0
                fps_actual = frame_idx / elapsed
                eta = (n_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
                print(f"  {frame_idx}/{n_frames} ({100*frame_idx/n_frames:.0f}%) "
                      f"— {fps_actual:.1f} fps, ETA {eta:.0f}s")

        write_proc.stdin.close()
        write_proc.wait()
        read_proc.terminate()
        read_proc.wait()

        if audio_tmp:
            Path(audio_tmp.name).unlink(missing_ok=True)

        elapsed = time.time() - t0
        print(f"Done! {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")
        return output_path

    def _compose_frame(self, frame_idx: int, video_frame: Image.Image) -> Image.Image:
        """Compose all panels into a single frame."""
        canvas = Image.new("RGB", (self.W, self.H), BG)

        # Place video
        vx, vy, vw, vh = self.video_rect
        vf = video_frame.resize((vw, vh), Image.LANCZOS)
        canvas.paste(vf, (vx, vy))

        # Apply video overlay if set
        if self.video_overlay:
            overlay = self.video_overlay.draw(frame_idx)
            if overlay.mode == "RGBA":
                region = canvas.crop((vx, vy, vx + vw, vy + vh)).convert("RGBA")
                composited = Image.alpha_composite(region, overlay)
                canvas.paste(composited.convert("RGB"), (vx, vy))
            else:
                canvas.paste(overlay, (vx, vy))

        # Draw each panel
        for panel, px, py in self.panels:
            panel_img = panel.draw(frame_idx)
            canvas.paste(panel_img, (px, py))

        return canvas
