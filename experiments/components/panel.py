"""Panel ABC — all renderer panels implement this interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Shared color palette
BG = (18, 18, 24)
TXT = (255, 255, 255)
TXT_DIM = (140, 140, 155)
ACCENT = (255, 220, 50)
C_ENERGY_LO = (50, 130, 200)
C_ENERGY_HI = (255, 80, 40)
C_COM = (255, 80, 80)
C_CYCLIC = (80, 220, 140)
C_BASS = (80, 220, 100)
C_PERC = (255, 140, 50)
C_HARM = (80, 160, 255)
SEG_COLORS = {
    "toprock":   (33, 150, 243),
    "footwork":  (76, 175, 80),
    "powermove": (220, 40, 40),
    "freeze":    (156, 39, 176),
}

# SMPL 22-joint skeleton
JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]

# 6 key tracked joints: (index, display_name, color)
KEY_JOINTS = [
    (15, "HEAD", (255, 220, 80)),
    (0,  "PELVIS", C_COM),
    (20, "L.WRIST", (200, 150, 255)),
    (21, "R.WRIST", (200, 150, 255)),
    (7,  "L.ANKLE", (100, 220, 180)),
    (8,  "R.ANKLE", (100, 220, 180)),
]

BONES = [
    (0,1),(0,2),(0,3),(1,4),(2,5),(3,6),(4,7),(5,8),(6,9),
    (7,10),(8,11),(9,12),(12,13),(12,14),(12,15),(13,16),(14,17),
    (16,18),(17,19),(18,20),(19,21),
]


class FontCache:
    """Lazy-loaded font cache shared across panels."""
    _fonts: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    @classmethod
    def get(cls, bold: bool = False, size: int = 14) -> ImageFont.FreeTypeFont:
        path = FONT_B if bold else FONT_R
        key = (path, size)
        if key not in cls._fonts:
            cls._fonts[key] = ImageFont.truetype(path, size)
        return cls._fonts[key]


class Panel(ABC):
    """Base class for all renderer panels."""

    def __init__(self, width: int, height: int, state):
        self.w = width
        self.h = height
        self.state = state  # WorldState instance

    def prerender(self) -> None:
        """Optional: compute static elements once before frame loop."""
        pass

    @abstractmethod
    def draw(self, frame_idx: int) -> Image.Image:
        """Draw this panel for the given frame. Returns RGB PIL Image."""
        ...

    def _blank(self) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        """Create a blank canvas for this panel."""
        img = Image.new("RGB", (self.w, self.h), BG)
        return img, ImageDraw.Draw(img)

    def _blank_rgba(self) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        """Create a blank RGBA canvas (for overlays)."""
        img = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        return img, ImageDraw.Draw(img)
