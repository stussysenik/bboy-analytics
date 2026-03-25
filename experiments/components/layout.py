"""Design tokens and grid layout builder for video analytics renderers.

Converts declarative Row/Col definitions into pixel-resolved Rects,
then wires them into RendererBase via add_panel / set_video_rect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Design tokens — single source of truth for colors, spacing, font sizes
# ---------------------------------------------------------------------------

TOKENS = {
    "gap": 4,   # px between rows/columns in grid layouts
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Col:
    """A column inside a row. *weight* sets relative width."""
    slot: str
    weight: float = 1.0
    is_video: bool = False


@dataclass
class Row:
    """A horizontal band of columns at a fixed pixel height."""
    height: int
    cols: list[Col] = field(default_factory=list)


@dataclass
class Rect:
    """Pixel-resolved rectangle for a named slot."""
    slot: str
    x: int
    y: int
    w: int
    h: int
    is_video: bool = False


# ---------------------------------------------------------------------------
# Grid computation — pure function, no side effects
# ---------------------------------------------------------------------------


def compute_grid(
    canvas_w: int,
    canvas_h: int,
    rows: list[Row],
    gap: int = TOKENS["gap"],
) -> list[Rect]:
    """Resolve Row/Col definitions into absolute pixel Rects.

    Gaps are placed *between* rows and *between* columns (not at edges).
    The last column in each row absorbs leftover pixels so the row fills
    exactly *canvas_w* with no rounding gap.
    """
    total_row_gap = gap * (len(rows) - 1) if len(rows) > 1 else 0
    usable_h = canvas_h - total_row_gap

    # Verify declared heights fit
    declared_h = sum(r.height for r in rows)
    if declared_h > usable_h:
        raise ValueError(
            f"Row heights ({declared_h}px) exceed usable canvas "
            f"({usable_h}px with {total_row_gap}px gaps)"
        )

    rects: list[Rect] = []
    cur_y = 0

    for row_idx, row in enumerate(rows):
        if row_idx > 0:
            cur_y += gap

        n_cols = len(row.cols)
        col_gap_total = gap * (n_cols - 1) if n_cols > 1 else 0
        usable_w = canvas_w - col_gap_total
        total_weight = sum(c.weight for c in row.cols)

        cur_x = 0
        for col_idx, col in enumerate(row.cols):
            if col_idx > 0:
                cur_x += gap

            is_last = col_idx == n_cols - 1
            if is_last:
                col_w = canvas_w - cur_x
            else:
                col_w = int(usable_w * col.weight / total_weight)

            rects.append(Rect(
                slot=col.slot,
                x=cur_x, y=cur_y,
                w=col_w, h=row.height,
                is_video=col.is_video,
            ))
            cur_x += col_w

        cur_y += row.height

    return rects


# ---------------------------------------------------------------------------
# Wiring helper — connects Rects to RendererBase
# ---------------------------------------------------------------------------


def apply_grid(
    renderer,
    rects: list[Rect],
    panel_map: dict,
    overlay_map: dict | None = None,
) -> None:
    """Place panels onto *renderer* according to resolved *rects*.

    For video slots (``is_video=True``), calls ``set_video_rect`` and
    optionally ``set_video_overlay`` from *overlay_map*.  All other
    slots are wired via ``add_panel``.
    """
    overlay_map = overlay_map or {}

    for rect in rects:
        if rect.is_video:
            renderer.set_video_rect(rect.x, rect.y, rect.w, rect.h)
            if rect.slot in overlay_map:
                renderer.set_video_overlay(overlay_map[rect.slot])
        else:
            if rect.slot in panel_map:
                renderer.add_panel(panel_map[rect.slot], rect.x, rect.y)
            else:
                print(f"Warning: slot '{rect.slot}' has no panel in panel_map")


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    layout = [
        Row(640, [Col("video", weight=3, is_video=True), Col("side", weight=1)]),
        Row(120, [Col("a"), Col("b"), Col("c")]),
    ]
    rects = compute_grid(1920, 1080, layout, gap=4)

    by_slot = {r.slot: r for r in rects}

    # Video row starts at y=0
    assert by_slot["video"].x == 0
    assert by_slot["video"].y == 0

    # Side panel abuts video with a gap
    assert by_slot["side"].x == by_slot["video"].w + 4

    # Last column in each row reaches canvas edge
    assert by_slot["side"].x + by_slot["side"].w == 1920
    assert by_slot["c"].x + by_slot["c"].w == 1920

    # Second row offset by first row height + gap
    assert by_slot["a"].y == 640 + 4

    # All bottom-row panels share the same height
    assert by_slot["a"].h == by_slot["b"].h == by_slot["c"].h == 120

    print("All assertions passed.")
    for r in rects:
        print(f"  {r.slot:>10}  ({r.x:4d}, {r.y:3d})  {r.w:4d}x{r.h}")
