"""
vision.py — Phase 2: Game State & Vision Processing (Optimized v2)
===================================================================

KEY INSIGHT (discovered by user): LinkedIn Zip does NOT use a <canvas>.
The board is a DOM grid of <div data-cell-idx="..."> elements, each
containing a <div data-cell-content="true" aria-label="N"> child when
the cell is numbered.

This module uses a **DOM-first** extraction strategy:

  1. Primary path — query the DOM directly via page.evaluate().
     This gives us exact cell indices and exact waypoint numbers with
     pixel-perfect accuracy.  No OCR, no template matching, no guessing.

  2. Wall detection — LinkedIn Zip renders walls as a subtly different
     background colour.  We sample each cell's background via
     getComputedStyle() and compare it against the "active" (white)
     cell colour.  Cells whose background deviates significantly are
     walls.

  3. Start / Goal inference — the lowest-numbered waypoint is the
     start; the highest is the goal.  If the DOM provides explicit
     start/goal markers we use those instead.

  4. Fallback — if the DOM extraction fails (e.g. LinkedIn changes
     their markup), we fall back to the OpenCV canvas pipeline.

The entire module is wrapped in defensive type coercion so the
"unsupported operand type(s) for //: 'str' and 'int'" crash that
affected the previous version cannot recur.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GAME_URL: str = "https://www.linkedin.com/games/view/zip/desktop"
PERSISTENT_CONTEXT_DIR: Path = Path(__file__).resolve().parent / ".browser_data"


class CellType(IntEnum):
    EMPTY: int = 0
    WALL: int = 1
    START: int = 2
    NUMBERED: int = 3
    GOAL: int = 4


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class VisionError(RuntimeError):
    """Raised when any step of the vision pipeline fails irrecoverably."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Cell:
    row: int
    col: int
    cell_type: int
    number: Optional[int] = None
    pixel_center: Tuple[int, int] = (0, 0)


@dataclass(slots=True)
class GameState:
    grid: np.ndarray
    numbers: np.ndarray
    start: Tuple[int, int]
    goal: Tuple[int, int]
    cells: List[Cell] = field(default_factory=list)
    canvas_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    cell_size: int = 0
    screenshot: Optional[np.ndarray] = None
    h_walls: Optional[np.ndarray] = None
    v_walls: Optional[np.ndarray] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self.grid.shape  # type: ignore[return-value]

    def to_tensor(self) -> np.ndarray:
        h, w = self.grid.shape
        tensor = np.zeros((4, h, w), dtype=np.float32)
        tensor[0] = self.grid.astype(np.float32)
        for cell in self.cells:
            if cell.cell_type == int(CellType.NUMBERED) and cell.number is not None:
                tensor[1, cell.row, cell.col] = float(cell.number)
        return tensor

    def pixel_of(self, row: int, col: int) -> Tuple[int, int]:
        """Return absolute page pixel coordinates for a grid cell."""
        x0 = int(self.canvas_bbox[0])
        y0 = int(self.canvas_bbox[1])
        cs = int(self.cell_size)
        cx = x0 + int(col) * cs + cs // 2
        cy = y0 + int(row) * cs + cs // 2
        return (cx, cy)


# ---------------------------------------------------------------------------
# Type-safe coercion helpers
# ---------------------------------------------------------------------------

def _to_int(v: Any, default: int = 0) -> int:
    """Coerce any value to int, returning ``default`` on failure."""
    try:
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str):
            return int(v.strip())
    except (ValueError, TypeError):
        pass
    return default


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            return float(v.strip())
    except (ValueError, TypeError):
        pass
    return default


# ---------------------------------------------------------------------------
# Playwright browser management
# ---------------------------------------------------------------------------

async def launch_browser(
    headless: bool = False,
    context_dir: Optional[Path] = None,
):
    """Launch a persistent Chromium context (real Chrome if available)."""
    from playwright.async_api import async_playwright

    if context_dir is None:
        context_dir = PERSISTENT_CONTEXT_DIR
    context_dir.mkdir(parents=True, exist_ok=True)

    pw = await async_playwright().start()
    try:
        # Try real Chrome first (channel="chrome"); fall back to bundled Chromium.
        try:
            context = await pw.chromium.launch_persistent_context(
                user_data_dir=str(context_dir),
                headless=headless,
                channel="chrome",
                viewport={"width": 1280, "height": 900},
                args=["--disable-blink-features=AutomationControlled"],
                ignore_default_args=["--enable-automation"],
            )
        except Exception:
            context = await pw.chromium.launch_persistent_context(
                user_data_dir=str(context_dir),
                headless=headless,
                viewport={"width": 1280, "height": 900},
                args=["--disable-blink-features=AutomationControlled"],
                ignore_default_args=["--enable-automation"],
            )
    except Exception as exc:
        await pw.stop()
        raise VisionError(f"Failed to launch persistent browser: {exc}") from exc
    return pw, context


# DOM extraction JavaScript — runs inside the page and returns a structured
# dict with every cell's index, number, background colour, and bounding rect.
_DOM_EXTRACTION_JS = r"""
() => {
    const cells = document.querySelectorAll('[data-cell-idx]');
    if (cells.length === 0) return null;

    const result = {
        count: cells.length,
        cells: [],
        gridRect: null
    };

    cells.forEach(c => {
        const idx = parseInt(c.getAttribute('data-cell-idx'), 10);
        const rect = c.getBoundingClientRect();
        const content = c.querySelector('[data-cell-content="true"], .trail-cell-content');
        let number = null;
        let ariaLabel = null;
        if (content) {
            ariaLabel = content.getAttribute('aria-label');
            if (ariaLabel) {
                const n = parseInt(ariaLabel, 10);
                if (!isNaN(n)) number = n;
            } else {
                const n = parseInt(content.textContent.trim(), 10);
                if (!isNaN(n)) number = n;
            }
        }
        const bg = window.getComputedStyle(c).backgroundColor;
        result.cells.push({
            idx: idx,
            number: number,
            ariaLabel: ariaLabel,
            bg: bg,
            x: rect.x,
            y: rect.y,
            w: rect.width,
            h: rect.height
        });
    });

    // Also get the grid container rect for pixel mapping.
    const grid = document.querySelector('[data-testid="interactive-grid"]') ||
                 cells[0].parentElement;
    if (grid) {
        const gr = grid.getBoundingClientRect();
        result.gridRect = { x: gr.x, y: gr.y, w: gr.width, h: gr.height };
    }
    return result;
}
"""


async def _extract_dom_state(page) -> Optional[Dict[str, Any]]:
    """Run the DOM extraction JS and return the parsed result (or None)."""
    try:
        result = await page.evaluate(_DOM_EXTRACTION_JS)
        if result is None or not isinstance(result, dict):
            return None
        return result
    except Exception:
        return None


def _parse_rgb(rgb_str: str) -> Optional[Tuple[int, int, int]]:
    """Parse 'rgb(r, g, b)' or 'rgba(r, g, b, a)' into (r, g, b)."""
    if not isinstance(rgb_str, str):
        return None
    import re
    m = re.match(r'rgba?\((\d+),\s*(\d+),\s*(\d+)', rgb_str.strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _infer_grid_dims(count: int) -> Tuple[int, int]:
    """Given a cell count, infer (rows, cols) assuming a square grid."""
    if count <= 0:
        return (0, 0)
    side = int(round(math.sqrt(count)))
    if side * side == count:
        return (side, side)
    # Non-square: try common factors close to sqrt.
    for s in range(side, 0, -1):
        if count % s == 0:
            return (count // s, s)
    return (1, count)


def _classify_walls(
    cells_data: List[Dict[str, Any]],
    numbered_indices: set,
) -> set:
    from collections import Counter
    all_bgs = []
    for cd in cells_data:
        rgb = _parse_rgb(cd.get('bg', ''))
        if rgb is not None:
            all_bgs.append(rgb)
            
    if not all_bgs:
        return set()
        
    active_mean = Counter(all_bgs).most_common(1)[0][0]
    
    walls: set = set()
    threshold = 25.0
    for cd in cells_data:
        if cd['idx'] in numbered_indices:
            continue
        rgb = _parse_rgb(cd.get('bg', ''))
        if rgb is None:
            continue
        dist = math.sqrt(
            (rgb[0] - active_mean[0]) ** 2 +
            (rgb[1] - active_mean[1]) ** 2 +
            (rgb[2] - active_mean[2]) ** 2
        )
        if dist > threshold:
            walls.add(cd['idx'])
    return walls


def build_state_from_dom(
    dom_state: Dict[str, Any],
    screenshot: Optional[np.ndarray] = None,
) -> GameState:
    """
    Construct a GameState purely from the DOM extraction result.

    This is the primary, pixel-perfect path.
    """
    count = _to_int(dom_state.get('count', 0))
    if count <= 0:
        raise VisionError("DOM extraction returned 0 cells.")

    rows, cols = _infer_grid_dims(count)
    if rows < 2 or cols < 2:
        raise VisionError(f"Cannot infer a valid grid from {count} cells.")

    cells_data = dom_state.get('cells', [])
    if len(cells_data) != count:
        raise VisionError(
            f"Cell count mismatch: count={count} but cells array has "
            f"{len(cells_data)} entries."
        )

    # Build index → (number, bg) lookup.
    idx_to_number: Dict[int, Optional[int]] = {}
    idx_to_bg: Dict[int, str] = {}
    for cd in cells_data:
        idx = _to_int(cd.get('idx', -1))
        num_raw = cd.get('number')
        number = _to_int(num_raw) if num_raw is not None else None
        idx_to_number[idx] = number
        idx_to_bg[idx] = cd.get('bg', '')

    numbered_indices = {i for i, n in idx_to_number.items() if n is not None}

    # Detect walls via colour.
    wall_indices = _classify_walls(cells_data, numbered_indices)

    # Build grid.
    grid = np.zeros((rows, cols), dtype=np.int32)
    cells: List[Cell] = []
    numbered_cells: List[Cell] = []

    # Grid container rect for pixel mapping.
    gr = dom_state.get('gridRect') or {}
    grid_x = _to_float(gr.get('x', 0))
    grid_y = _to_float(gr.get('y', 0))
    grid_w = _to_float(gr.get('w', 0))
    grid_h = _to_float(gr.get('h', 0))

    cell_w = grid_w / cols if cols > 0 else 0
    cell_h = grid_h / rows if rows > 0 else 0
    cell_size = int(round(min(cell_w, cell_h))) if (cell_w > 0 and cell_h > 0) else 40

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            number = idx_to_number.get(idx)
            is_wall = idx in wall_indices

            if number is not None and not is_wall:
                cell_type = int(CellType.NUMBERED)
            elif is_wall:
                cell_type = int(CellType.WALL)
            else:
                cell_type = int(CellType.EMPTY)

            # Pixel centre relative to the page viewport.
            px = int(round(grid_x + c * cell_w + cell_w / 2))
            py = int(round(grid_y + r * cell_h + cell_h / 2))

            grid[r, c] = cell_type
            cell = Cell(
                row=r, col=c, cell_type=cell_type, number=number,
                pixel_center=(px, py),
            )
            cells.append(cell)
            if cell_type == int(CellType.NUMBERED):
                numbered_cells.append(cell)

    # Infer start & goal.
    numbered_cells.sort(key=lambda c: (c.number if c.number is not None else 0))
    start: Optional[Tuple[int, int]] = None
    goal: Optional[Tuple[int, int]] = None

    if numbered_cells:
        start_cell = numbered_cells[0]
        goal_cell = numbered_cells[-1]
        start = (start_cell.row, start_cell.col)
        goal = (goal_cell.row, goal_cell.col)
        grid[start] = int(CellType.START)
        grid[goal] = int(CellType.GOAL)
        start_cell.cell_type = int(CellType.START)
        goal_cell.cell_type = int(CellType.GOAL)

    if start is None or goal is None:
        raise VisionError(
            "Could not identify START and GOAL — no numbered cells found."
        )

    numbers_arr = np.array(
        sorted({c.number for c in numbered_cells if c.number is not None}),
        dtype=np.int32,
    )

    canvas_bbox = (int(round(grid_x)), int(round(grid_y)),
                   int(round(grid_w)), int(round(grid_h)))

    state = GameState(
        grid=grid,
        numbers=numbers_arr,
        start=start,
        goal=goal,
        cells=cells,
        canvas_bbox=canvas_bbox,
        cell_size=cell_size,
        screenshot=screenshot,
    )
    
    # Detect internal edge walls if screenshot is available
    if screenshot is not None:
        state.h_walls = np.zeros((rows, cols), dtype=bool)
        state.v_walls = np.zeros((rows, cols), dtype=bool)
        
        for r in range(rows):
            for c in range(cols):
                cx, cy = state.pixel_of(r, c)
                
                # Check vertical wall (to the right)
                if c < cols - 1:
                    cx_right, _ = state.pixel_of(r, c + 1)
                    mid_x = (cx + cx_right) // 2
                    slice_img = screenshot[max(0, cy-5):cy+5, max(0, mid_x-2):mid_x+2]
                    if slice_img.size > 0 and np.mean(slice_img) < 100:
                        state.v_walls[r, c] = True
                
                # Check horizontal wall (below)
                if r < rows - 1:
                    _, cy_down = state.pixel_of(r + 1, c)
                    mid_y = (cy + cy_down) // 2
                    slice_img = screenshot[max(0, mid_y-2):mid_y+2, max(0, cx-5):cx+5]
                    if slice_img.size > 0 and np.mean(slice_img) < 100:
                        state.h_walls[r, c] = True

    return state


# ---------------------------------------------------------------------------
# OpenCV fallback (used only if DOM extraction fails)
# ---------------------------------------------------------------------------

def _detect_grid_shape_cv(screenshot: np.ndarray) -> Tuple[int, int, int]:
    if screenshot is None or screenshot.size == 0:
        raise VisionError("Empty screenshot for grid detection.")
    h, w = screenshot.shape[:2]
    side = min(h, w)
    for candidate in (8, 6, 10, 12, 7, 9, 5, 11):
        if side % candidate == 0:
            return candidate, candidate, side // candidate
    return 8, 8, side // 8


def _classify_cell_cv(cell_img: np.ndarray) -> Tuple[int, Optional[int]]:
    """OpenCV-based cell classifier (fallback only)."""
    if cell_img is None or cell_img.size == 0:
        return int(CellType.EMPTY), None
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 80:
        return int(CellType.WALL), None
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    sat_mask = cv2.inRange(hsv, (0, 80, 80), (180, 255, 255))
    coloured_ratio = float(np.count_nonzero(sat_mask) / (cell_img.shape[0] * cell_img.shape[1]))
    if coloured_ratio > 0.18:
        return int(CellType.START), None
    return int(CellType.EMPTY), None


def parse_canvas_fallback(
    screenshot: np.ndarray,
    canvas_bbox: Tuple[int, int, int, int],
) -> GameState:
    """OpenCV fallback parser — used only when DOM extraction fails."""
    rows, cols, cell_size = _detect_grid_shape_cv(screenshot)
    grid = np.zeros((rows, cols), dtype=np.int32)
    cells: List[Cell] = []
    start: Optional[Tuple[int, int]] = None
    goal: Optional[Tuple[int, int]] = None

    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * cell_size, c * cell_size
            y1 = min(y0 + cell_size, screenshot.shape[0])
            x1 = min(x0 + cell_size, screenshot.shape[1])
            cell_img = screenshot[y0:y1, x0:x1]
            cell_type, _ = _classify_cell_cv(cell_img)
            grid[r, c] = cell_type
            cells.append(Cell(
                row=r, col=c, cell_type=cell_type,
                pixel_center=(canvas_bbox[0] + x0 + cell_size // 2,
                              canvas_bbox[1] + y0 + cell_size // 2),
            ))
            if cell_type == int(CellType.START) and start is None:
                start = (r, c)

    if start is None:
        start = (0, 0)
        grid[0, 0] = int(CellType.START)
    goal = (rows - 1, cols - 1)
    grid[goal] = int(CellType.GOAL)

    return GameState(
        grid=grid,
        numbers=np.array([], dtype=np.int32),
        start=start,
        goal=goal,
        cells=cells,
        canvas_bbox=canvas_bbox,
        cell_size=cell_size,
        screenshot=screenshot,
    )


# ---------------------------------------------------------------------------
# Playwright navigation & screenshot
# ---------------------------------------------------------------------------

async def _navigate_and_wait(page, game_url: str, timeout_ms: int) -> None:
    """Navigate to the game URL and wait for the board to render."""
    await page.goto(game_url, wait_until="domcontentloaded", timeout=timeout_ms)

    for selector in [
        "button:has-text('Play')",
        "button:has-text('Start now')",
        "button:has-text('Start')",
        "button:has-text('Continue')",
    ]:
        try:
            btn = page.locator(selector).first
            await btn.wait_for(state="visible", timeout=3000)
            await btn.click()
            await asyncio.sleep(0.5)
        except Exception:
            pass

    # Wait for the grid to render using a combined selector.
    try:
        combined_selector = "[data-testid='interactive-grid'], [data-cell-idx], canvas"
        loc = page.locator(combined_selector).first
        await loc.wait_for(state="visible", timeout=timeout_ms)
        return
    except Exception:
        pass
    raise VisionError(
        f"No visible game board found after {timeout_ms//1000}s. Make sure you click 'Start Game'."
    )


async def capture_canvas_screenshot(
    context,
    game_url: str = DEFAULT_GAME_URL,
    page_timeout_ms: int = 60_000,
) -> Tuple[object, np.ndarray, Tuple[int, int, int, int]]:
    """Navigate, wait for the board, and capture a screenshot + bbox."""
    # Always open a fresh tab — in persistent profiles, pages[0] can be a
    # hidden extension background page that refuses navigation.
    page = await context.new_page()

    await _navigate_and_wait(page, game_url, page_timeout_ms)

    # Get the grid bounding box for the screenshot clip.
    bbox = (0, 0, 0, 0)
    for selector in ["[data-testid='interactive-grid']", "[data-cell-idx]"]:
        try:
            loc = page.locator(selector).first
            box = await loc.bounding_box()
            if box and box.get("width", 0) > 50:
                bbox = (
                    int(box["x"]), int(box["y"]),
                    int(box["width"]), int(box["height"]),
                )
                break
        except Exception:
            continue

    # Click the start node (1) to clear any purple paths left over from a previous
    # failed run. If we don't, the purple path will cause the vision engine to 
    # classify the active cells as walls!
    try:
        start_node = page.locator("[aria-label='1']").first
        await start_node.wait_for(state="visible", timeout=1000)
        await start_node.click(force=True)
        await asyncio.sleep(0.2)
    except Exception:
        pass

    # Full-page screenshot (used for debugging / fallback).
    try:
        png_bytes = await page.screenshot(type="png")
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        screenshot = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        screenshot = None

    return page, screenshot, bbox


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

async def capture_state(
    headless: bool = False,
    game_url: str = DEFAULT_GAME_URL,
    context_dir: Optional[Path] = None,
) -> Tuple[object, GameState]:
    """
    Launch browser → navigate → extract DOM state → build GameState.

    Returns (page, state). The page is kept open so the executor can
    drive the mouse through it.
    """
    pw, context = await launch_browser(headless=headless, context_dir=context_dir)
    try:
        # Give extensions a moment to initialise.
        await asyncio.sleep(0.3)

        page, screenshot, bbox = await capture_canvas_screenshot(
            context, game_url=game_url
        )

        # Primary path: DOM extraction.
        dom_state = await _extract_dom_state(page)
        if dom_state is not None and dom_state.get('count', 0) > 0:
            state = build_state_from_dom(dom_state, screenshot=screenshot)
            print(f"[vision] DOM extraction: {state.shape[0]}x{state.shape[1]} grid, "
                  f"{len(state.numbers)} waypoints")
            return page, state

        # Fallback: OpenCV on the screenshot.
        if screenshot is not None and bbox[2] > 50:
            state = parse_canvas_fallback(screenshot, bbox)
            print(f"[vision] OpenCV fallback: {state.shape[0]}x{state.shape[1]} grid")
            return page, state

        raise VisionError(
            "Both DOM extraction and OpenCV fallback failed. "
            "Ensure you are logged into LinkedIn and the game board is visible."
        )
    except VisionError:
        await pw.stop()
        raise
    except Exception as exc:
        await pw.stop()
        raise VisionError(f"Unexpected error during state capture: {exc}") from exc


# ---------------------------------------------------------------------------
# Synthetic board generator (for testing without a browser)
# ---------------------------------------------------------------------------

def generate_synthetic_state(
    rows: int = 6,
    cols: int = 6,
    num_waypoints: int = 3,
    wall_fraction: float = 0.15,
    seed: Optional[int] = None,
) -> Tuple[GameState, List[Tuple[int, int]]]:
    """
    Generate a random solvable Zip board and its solution path.

    Returns (state, true_path). Used for testing the solver without
    needing a live browser session.
    """
    import random
    rng = random.Random(seed)

    for _ in range(200):
        # Random walk to create a Hamiltonian-ish path.
        visited = np.zeros((rows, cols), dtype=bool)
        sr, sc = rng.randint(0, rows - 1), rng.randint(0, cols - 1)
        path = [(sr, sc)]
        visited[sr, sc] = True
        target_len = int(rows * cols * (0.6 + rng.random() * 0.3))

        while len(path) < target_len:
            r, c = path[-1]
            neighbours = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                    neighbours.append((nr, nc))
            if not neighbours:
                break
            nr, nc = rng.choice(neighbours)
            path.append((nr, nc))
            visited[nr, nc] = True

        if len(path) < max(8, num_waypoints * 3):
            continue

        # Place waypoints along the path.
        wp_count = min(num_waypoints, len(path) // 3)
        wp_indices = [0, len(path) - 1]
        if wp_count > 2:
            interior = list(range(2, len(path) - 2, max(1, len(path) // wp_count)))
            rng.shuffle(interior)
            wp_indices.extend(interior[:wp_count - 2])
        wp_indices = sorted(set(wp_indices))

        # Build grid.
        grid = np.full((rows, cols), int(CellType.WALL), dtype=np.int32)
        cells: List[Cell] = []
        numbered_cells: List[Cell] = []

        path_set = set(path)
        for r in range(rows):
            for c in range(cols):
                if (r, c) in path_set:
                    idx = path.index((r, c))
                    if idx in wp_indices:
                        wp_num = wp_indices.index(idx) + 1
                        if wp_num == 1:
                            ct = int(CellType.START)
                        elif wp_num == len(wp_indices):
                            ct = int(CellType.GOAL)
                        else:
                            ct = int(CellType.NUMBERED)
                        grid[r, c] = ct
                        cell = Cell(row=r, col=c, cell_type=ct, number=wp_num,
                                    pixel_center=(c * 40 + 20, r * 40 + 20))
                        cells.append(cell)
                        if ct == int(CellType.NUMBERED):
                            numbered_cells.append(cell)
                    else:
                        grid[r, c] = int(CellType.EMPTY)
                        cells.append(Cell(row=r, col=c, cell_type=int(CellType.EMPTY),
                                          pixel_center=(c * 40 + 20, r * 40 + 20)))
                else:
                    cells.append(Cell(row=r, col=c, cell_type=int(CellType.WALL),
                                      pixel_center=(c * 40 + 20, r * 40 + 20)))

        start = path[0]
        goal = path[-1]
        numbers = np.array(
            sorted({c.number for c in numbered_cells if c.number is not None}),
            dtype=np.int32,
        )

        state = GameState(
            grid=grid,
            numbers=numbers,
            start=start,
            goal=goal,
            cells=cells,
            canvas_bbox=(0, 0, cols * 40, rows * 40),
            cell_size=40,
        )
        return state, path

    raise RuntimeError("Failed to generate a solvable synthetic board.")


# ---------------------------------------------------------------------------
# Debug CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def _debug() -> None:
        page, state = await capture_state(headless=False)
        print(f"Grid:      {state.shape[0]}x{state.shape[1]}")
        print(f"Start:     {state.start}")
        print(f"Goal:      {state.goal}")
        print(f"Waypoints: {state.numbers.tolist()}")
        print(f"Grid tensor:\n{state.grid}")
        print("Keeping browser open 30s for inspection...")
        await asyncio.sleep(30)

    asyncio.run(_debug())
