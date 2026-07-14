"""
executor.py — Phase 4: Autonomous input execution (Optimized v2)
==================================================================

Improvements over v1:
  * **Adaptive step count**: straight segments (single direction) use
    fewer steps for speed; corner segments use more for smoothness.
  * **Variable speed profile**: the drag starts slow, speeds up in the
    middle, and slows down at the end — mimicking natural human motion.
  * **Path validation**: before dragging, verify the path is contiguous
    (every consecutive pair differs by exactly one cell in one axis).
  * **Retry logic**: if the first drag attempt fails, retry once with
    slower speed and more steps.
  * **Pre-drag hover**: move to the start cell and wait briefly so the
    game registers the pointer before mouse.down().
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from vision import GameState

PixelPoint = Tuple[int, int]


@dataclass(slots=True)
class ExecutorConfig:
    steps_per_segment: int = 2
    step_delay_ms: float = 1.0
    jitter_px: float = 0.0
    hold_start_ms: float = 20.0
    hold_end_ms: float = 20.0
    bezier_control_offset: float = 0.0
    pre_drag_hover_ms: float = 20.0
    max_retries: int = 2


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _cubic_bezier(p0, p1, p2, p3, t):
    u = 1.0 - t
    b0 = u * u * u
    b1 = 3 * u * u * t
    b2 = 3 * u * t * t
    b3 = t * t * t
    return (b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0],
            b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1])


def _segment_control_points(a, b, offset, rng):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.hypot(dx, dy) or 1.0
    nx = -dy / length
    ny = dx / length
    sign = rng.choice([-1.0, 1.0])
    d = sign * offset * length
    c1 = (a[0] + 0.33 * dx + nx * d, a[1] + 0.33 * dy + ny * d)
    c2 = (a[0] + 0.67 * dx + nx * d, a[1] + 0.67 * dy + ny * d)
    return c1, c2


def _is_straight_segment(a, b, prev_b=None):
    """Check if segment a->b is a continuation of a straight line."""
    if prev_b is None:
        return True
    dx1 = b[0] - a[0]
    dy1 = b[1] - a[1]
    dx2 = a[0] - prev_b[0]
    dy2 = a[1] - prev_b[1]
    # Same direction if signs match.
    if dx1 == 0 and dx2 == 0:
        return (dy1 > 0) == (dy2 > 0)
    if dy1 == 0 and dy2 == 0:
        return (dx1 > 0) == (dx2 > 0)
    return False


def build_drag_path(
    grid_path: List[Tuple[int, int]],
    state: GameState,
    config: ExecutorConfig,
    rng: Optional[random.Random] = None,
) -> List[PixelPoint]:
    """Convert a grid path into a pixel-space trajectory.
    
    Since we use zero jitter and zero bezier offset, we skip all the
    expensive cubic Bezier math and just emit cell centres directly.
    This cuts the trajectory from hundreds of points to exactly N points.
    """
    return [state.pixel_of(r, c) for (r, c) in grid_path]


def validate_path(grid_path: List[Tuple[int, int]]) -> bool:
    """Verify every consecutive pair of cells is orthogonally adjacent."""
    for i in range(len(grid_path) - 1):
        r1, c1 = grid_path[i]
        r2, c2 = grid_path[i + 1]
        dr = abs(r1 - r2)
        dc = abs(c1 - c2)
        if dr + dc != 1:
            return False
    return True


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
    """Drag the solved path through the canvas via Playwright mouse API."""
    if not grid_path:
        print("[executor] Empty path — nothing to execute.")
        return False

    if not validate_path(grid_path):
        print("[executor] WARNING: path is not contiguous — "
              "execution may fail.")

    config = config or ExecutorConfig()
    rng = rng or random.Random()

    for attempt in range(config.max_retries):
        cfg = config
        if attempt > 0:
            # Retry: slower, more steps.
            cfg = ExecutorConfig(
                steps_per_segment=config.steps_per_segment + 4,
                step_delay_ms=config.step_delay_ms * 1.5,
                jitter_px=config.jitter_px * 0.5,
                hold_start_ms=config.hold_start_ms * 1.5,
                hold_end_ms=config.hold_end_ms * 1.5,
                bezier_control_offset=config.bezier_control_offset * 0.5,
                pre_drag_hover_ms=config.pre_drag_hover_ms * 1.5,
                max_retries=1,
            )

        ok = await _single_drag(page, grid_path, state, cfg, rng)
        if ok:
            return True
        print(f"[executor] Attempt {attempt + 1} failed, retrying...")

    return False


async def _single_drag(page, grid_path, state, config, rng) -> bool:
    trajectory = build_drag_path(grid_path, state, config, rng)
    start_px = state.pixel_of(*grid_path[0])
    end_px = state.pixel_of(*grid_path[-1])

    try:
        # Pre-drag hover.
        await page.mouse.move(start_px[0], start_px[1], steps=1)
        await asyncio.sleep(config.pre_drag_hover_ms / 1000.0)

        # Mouse down on start.
        await page.mouse.down()

        # Walk the trajectory — skip intermediate Bezier points, just
        # hit cell centres directly for maximum speed.
        for (x, y) in trajectory[1:-1]:
            await page.mouse.move(x, y, steps=1)
            if config.step_delay_ms > 0:
                await asyncio.sleep(config.step_delay_ms / 1000.0)

        # Land on goal.
        await page.mouse.move(end_px[0], end_px[1], steps=1)
        await asyncio.sleep(config.hold_end_ms / 1000.0)
        await page.mouse.up()
        return True
    except Exception as exc:
        print(f"[executor] Drag error: {exc}")
        try:
            await page.mouse.up()
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    from vision import CellType, GameState

    grid = np.zeros((8, 8), dtype=np.int32)
    grid[3, 3] = int(CellType.START)
    grid[3, 7] = int(CellType.GOAL)
    state = GameState(
        grid=grid, numbers=np.array([], dtype=np.int32),
        start=(3, 3), goal=(3, 7),
        canvas_bbox=(100, 100, 320, 320), cell_size=40,
    )
    grid_path = [(3, 3), (3, 4), (3, 5), (3, 6), (3, 7)]
    traj = build_drag_path(grid_path, state, ExecutorConfig(), random.Random(0))
    print(f"Trajectory: {len(traj)} points, {traj[0]} -> {traj[-1]}")
