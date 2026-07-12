"""
executor.py
===========
Phase 4 - Autonomous input execution.

Translates the solved grid path back into canvas pixel coordinates and
drives the solution through Playwright's ``page.mouse`` API. The drag is
animated with cubic Bezier interpolation between consecutive cells and
small per-segment random jitter, so the motion looks organic rather than
machine-straight.
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from vision import GameState

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

PixelPoint = Tuple[int, int]


@dataclass(slots=True)
class ExecutorConfig:
    """
    Tunable parameters for the drag animation.

    Attributes
    ----------
    steps_per_segment : int
        Number of intermediate mouse moves between two adjacent cells.
        Higher = smoother but slower.
    step_delay_ms : float
        Delay between two consecutive mouse moves, in milliseconds.
    jitter_px : float
        Maximum perpendicular jitter in pixels, applied per intermediate
        sample. Keeps the curve from looking perfectly straight.
    hold_start_ms : float
        How long to hold the mouse button on the start cell before moving.
    hold_end_ms : float
        How long to hold on the goal cell before releasing.
    bezier_control_offset : float
        Distance of the Bezier control points from the straight line,
        expressed as a fraction of the segment length. 0 = straight line,
        0.2 = noticeable curve.
    """
    steps_per_segment: int = 16
    step_delay_ms: float = 8.0
    jitter_px: float = 2.5
    hold_start_ms: float = 120.0
    hold_end_ms: float = 200.0
    bezier_control_offset: float = 0.18


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _cubic_bezier(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    t: float,
) -> Tuple[float, float]:
    """Evaluate the cubic Bezier curve at parameter ``t`` in [0, 1]."""
    u = 1.0 - t
    b0 = u * u * u
    b1 = 3 * u * u * t
    b2 = 3 * u * t * t
    b3 = t * t * t
    x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
    y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
    return (x, y)


def _segment_control_points(
    a: PixelPoint,
    b: PixelPoint,
    offset: float,
    rng: random.Random,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute cubic Bezier control points for a segment from ``a`` to ``b``.

    The control points are placed along the perpendicular to the segment,
    offset by ``offset * |b - a|``. The sign of the offset is randomised so
    consecutive segments don't all curve the same way.
    """
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    length = math.hypot(dx, dy) or 1.0
    # Perpendicular unit vector.
    nx = -dy / length
    ny = dx / length
    sign = rng.choice([-1.0, 1.0])
    d = sign * offset * length

    c1 = (ax + 0.33 * dx + nx * d, ay + 0.33 * dy + ny * d)
    c2 = (ax + 0.67 * dx + nx * d, ay + 0.67 * dy + ny * d)
    return c1, c2


def build_drag_path(
    grid_path: List[Tuple[int, int]],
    state: GameState,
    config: ExecutorConfig,
    rng: Optional[random.Random] = None,
) -> List[PixelPoint]:
    """
    Convert a grid path into a smooth pixel-space drag trajectory.

    Parameters
    ----------
    grid_path : list of (row, col)
        Solved path from the solver.
    state : GameState
        Provides the canvas bounding box and cell size used to map grid
        coordinates to canvas pixels.
    config : ExecutorConfig
        Animation parameters.
    rng : random.Random, optional
        Source of randomness for jitter/control-point sign. Deterministic
        when fixed.

    Returns
    -------
    list of (x, y) pixel points inside the page viewport coordinate system.
    """
    rng = rng or random.Random()

    # Convert grid cells to canvas-centre pixels.
    centres: List[PixelPoint] = [state.pixel_of(r, c) for (r, c) in grid_path]
    if not centres:
        return []

    trajectory: List[PixelPoint] = [centres[0]]

    for i in range(len(centres) - 1):
        a = centres[i]
        b = centres[i + 1]
        c1, c2 = _segment_control_points(a, b, config.bezier_control_offset, rng)

        for s in range(1, config.steps_per_segment + 1):
            t = s / config.steps_per_segment
            x, y = _cubic_bezier(a, c1, c2, b, t)

            # Add a small randomised jitter that shrinks to 0 at the
            # endpoints of each segment so we don't perturb the cell centres.
            taper = math.sin(math.pi * t)  # 0 -> 1 -> 0
            jx = rng.uniform(-config.jitter_px, config.jitter_px) * taper
            jy = rng.uniform(-config.jitter_px, config.jitter_px) * taper
            trajectory.append((int(round(x + jx)), int(round(y + jy))))

    return trajectory


# ---------------------------------------------------------------------------
# Playwright execution
# ---------------------------------------------------------------------------

async def execute_path(
    page,
    grid_path: List[Tuple[int, int]],
    state: GameState,
    config: Optional[ExecutorConfig] = None,
    rng: Optional[random.Random] = None,
) -> bool:
    """
    Drive the solved path through Playwright's mouse API.

    The motion is:
        1. Move to the start cell centre.
        2. mouse.down()
        3. Walk through every intermediate Bezier sample.
        4. mouse.up() on the goal cell centre.

    Returns True on success, False on failure (with the error printed).
    """
    if not grid_path:
        print("[executor] Empty path - nothing to execute.")
        return False

    config = config or ExecutorConfig()
    rng = rng or random.Random()
    trajectory = build_drag_path(grid_path, state, config, rng)

    start_px = state.pixel_of(*grid_path[0])
    end_px = state.pixel_of(*grid_path[-1])

    try:
        # Move to start without holding the button.
        await page.mouse.move(start_px[0], start_px[1], steps=10)
        await asyncio.sleep(config.hold_start_ms / 1000.0)

        # Press.
        await page.mouse.down()

        # Drag through the Bezier trajectory.
        for (x, y) in trajectory[1:-1]:
            await page.mouse.move(x, y, steps=1)
            if config.step_delay_ms > 0:
                await asyncio.sleep(config.step_delay_ms / 1000.0)

        # Land exactly on the goal centre.
        await page.mouse.move(end_px[0], end_px[1], steps=5)
        await asyncio.sleep(config.hold_end_ms / 1000.0)

        # Release.
        await page.mouse.up()
        return True
    except Exception as exc:
        print(f"[executor] ERROR during drag: {exc}")
        # Best-effort cleanup: release the button if it's still held.
        try:
            await page.mouse.up()
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick visual sanity check of the trajectory shape on a fake 8x8 board.
    import numpy as np
    from vision import CellType, GameState

    grid = np.zeros((8, 8), dtype=np.int32)
    grid[3, 3] = int(CellType.START)
    grid[3, 7] = int(CellType.GOAL)
    state = GameState(
        grid=grid,
        numbers=np.array([], dtype=np.int32),
        start=(3, 3),
        goal=(3, 7),
        canvas_bbox=(100, 100, 320, 320),
        cell_size=40,
    )
    grid_path = [(3, 3), (3, 4), (3, 5), (3, 6), (3, 7)]
    cfg = ExecutorConfig()
    traj = build_drag_path(grid_path, state, cfg, random.Random(0))
    print(f"Trajectory length: {len(traj)} points")
    print(f"First: {traj[0]}  Last: {traj[-1]}")
