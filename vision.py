"""
vision.py
=========
Phase 2 - Game State & Vision Processing.

This module is responsible for turning the raw LinkedIn Zip game canvas into a
structured NumPy tensor the solver can reason about. The pipeline is:

    1. Launch Playwright with a *persistent* browser context so the user's
       LinkedIn login cookies survive between sessions.
    2. Navigate to the Zip game URL.
    3. Locate the ``<canvas>`` element and capture its bounding box plus a
       screenshot.
    4. Run an OpenCV pipeline that:
         * Converts the screenshot to grayscale.
         * Detects the grid cells using adaptive thresholding + contour
           analysis.
         * Classifies each cell as WALL / EMPTY / START / NUMBERED using
           colour and template-matching heuristics.
         * Recovers the numeric labels for numbered waypoints in the order
           they must be visited (1, 2, 3, ...).
    5. Pack everything into a ``GameState`` dataclass containing the grid
       tensor, cell->pixel mapping and metadata.

The module is intentionally defensive: every step wraps OpenCV/Playwright
calls in try/except and raises ``VisionError`` with a precise message so the
caller can decide whether to retry or abort.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GAME_URL: str = "https://www.linkedin.com/games/zip"
PERSISTENT_CONTEXT_DIR: Path = Path(__file__).resolve().parent / ".browser_data"

# Cell type encoding - this is the channel-0 value of the grid tensor.
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
    """A single grid cell with both grid and pixel coordinates."""
    row: int
    col: int
    cell_type: int            # CellType enum value
    number: Optional[int] = None  # only set for NUMBERED cells
    pixel_center: Tuple[int, int] = (0, 0)  # (x, y) inside the canvas


@dataclass(slots=True)
class GameState:
    """Structured representation of the parsed Zip board."""
    grid: np.ndarray              # shape (H, W), int32, values from CellType
    numbers: np.ndarray           # shape (K,), int32 - the numbered waypoint
                                  # values found on the board, sorted ascending
    start: Tuple[int, int]        # (row, col) of the start cell
    goal: Tuple[int, int]         # (row, col) of the goal cell
    cells: List[Cell] = field(default_factory=list)
    canvas_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    cell_size: int = 0            # pixels per cell side
    screenshot: Optional[np.ndarray] = None  # BGR screenshot for debugging

    @property
    def shape(self) -> Tuple[int, int]:
        return self.grid.shape

    def to_tensor(self) -> np.ndarray:
        """
        Convert the state to a multi-channel tensor suitable for the CNN.

        Channels
        --------
        0 : cell type (one-hot encoded as integer per CellType)
        1 : number value (0 for non-numbered cells, otherwise the waypoint id)
        2 : visited mask (all zero here - the solver fills it in as it walks)
        3 : current position mask (all zero here - the solver sets it)
        """
        h, w = self.grid.shape
        tensor = np.zeros((4, h, w), dtype=np.float32)
        tensor[0] = self.grid.astype(np.float32)
        for cell in self.cells:
            if cell.cell_type == CellType.NUMBERED and cell.number is not None:
                tensor[1, cell.row, cell.col] = float(cell.number)
        return tensor

    def pixel_of(self, row: int, col: int) -> Tuple[int, int]:
        """Return the absolute canvas pixel coordinates for a grid cell."""
        x0, y0, _, _ = self.canvas_bbox
        cx = x0 + col * self.cell_size + self.cell_size // 2
        cy = y0 + row * self.cell_size + self.cell_size // 2
        return (cx, cy)


# ---------------------------------------------------------------------------
# Playwright browser management
# ---------------------------------------------------------------------------

async def launch_browser(
    headless: bool = False,
    context_dir: Optional[Path] = None,
):
    """
    Launch a persistent Chromium context using a dedicated profile.
    """
    from playwright.async_api import async_playwright
    import os

    if context_dir is None:
        context_dir = PERSISTENT_CONTEXT_DIR

    context_dir.mkdir(parents=True, exist_ok=True)
    pw = await async_playwright().start()
    try:
        context = await pw.chromium.launch_persistent_context(
            user_data_dir=str(context_dir),
            headless=headless,
            channel="chrome",
            viewport={"width": 1280, "height": 900},
            args=["--disable-blink-features=AutomationControlled"],
            ignore_default_args=["--enable-automation"],
        )
    except Exception as exc:
        await pw.stop()
        raise VisionError(f"Failed to launch persistent Chromium: {exc}") from exc
    return pw, context


async def capture_canvas_screenshot(
    context,
    game_url: str = DEFAULT_GAME_URL,
    page_timeout_ms: int = 60_000,
) -> Tuple[object, np.ndarray, Tuple[int, int, int, int]]:
    """
    Navigate to the Zip game and return (page, screenshot_bgr, canvas_bbox).

    canvas_bbox is (x, y, width, height) relative to the **page** (viewport)
    coordinate system - the same coordinate system used by Playwright's
    ``page.mouse`` API, which is exactly what the executor needs.
    """
    import asyncio
    # Give Chrome's extensions a moment to initialize before we try to drive it
    await asyncio.sleep(3.0)
    
    # Always create a new tab. In a persistent profile, context.pages[0] can sometimes 
    # be a hidden extension background page which refuses to navigate!
    page = await context.new_page()
    try:
        await page.goto(game_url, wait_until="domcontentloaded",
                        timeout=page_timeout_ms)
    except Exception as exc:
        raise VisionError(f"Navigation to {game_url} failed: {exc}") from exc

    # Attempt to click a 'Play' or 'Start now' button if the game has a splash screen
    try:
        play_btn = page.locator("button:has-text('Play'), button:has-text('Start now'), button:has-text('Start')").first
        await play_btn.wait_for(state="visible", timeout=5000)
        await play_btn.click()
    except Exception:
        pass

    # Wait for the Zip board container to render.
    try:
        grid_locator = page.locator("[data-testid='interactive-grid']")
        await grid_locator.wait_for(state="visible", timeout=page_timeout_ms)
    except Exception as exc:
        raise VisionError(
            "No visible game board found on the page. "
            "Are you logged into LinkedIn?"
        ) from exc

    box = await grid_locator.bounding_box()
    if box is None:
        raise VisionError("Grid bounding box could not be read.")
    bbox = (int(box["x"]), int(box["y"]), int(box["width"]), int(box["height"]))

    # Capture a cropped screenshot of the game board.
    png_bytes = await grid_locator.screenshot(type="png")
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    screenshot = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if screenshot is None:
        raise VisionError("Failed to decode the canvas screenshot.")
    return page, screenshot, bbox


# ---------------------------------------------------------------------------
# OpenCV grid parsing
# ---------------------------------------------------------------------------

def _detect_grid_shape(
    screenshot: np.ndarray,
) -> Tuple[int, int, int]:
    """
    Infer (rows, cols, cell_size) from the canvas screenshot.

    Strategy
    --------
    Convert to grayscale, run adaptive thresholding to highlight the grid
    lines, then use the Hough line transform to find dominant vertical and
    horizontal line spacings. The greatest common divisor of those spacings
    is the cell size.
    """
    if screenshot is None or screenshot.size == 0:
        raise VisionError("Empty screenshot passed to grid detector.")

    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=10,
    )

    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80,
        minLineLength=int(screenshot.shape[1] * 0.4),
        maxLineGap=10,
    )

    if lines is None:
        # Fall back to a square-grid heuristic: assume the board is square
        # and subdivide equally.
        h, w = screenshot.shape[:2]
        side = min(h, w)
        # Zip boards on LinkedIn are typically 8x8 but can be 6x6 / 10x10.
        # Use the side length to guess.
        for candidate in (8, 6, 10, 12, 7, 9):
            if side % candidate == 0:
                return candidate, candidate, side // candidate
        # Last resort: assume 8x8.
        cell = side // 8
        return 8, 8, cell

    xs: List[int] = []
    ys: List[int] = []
    for ln in lines.reshape(-1, 4):
        x1, y1, x2, y2 = ln
        if abs(y1 - y2) <= 2 and abs(x2 - x1) > screenshot.shape[1] * 0.3:
            ys.append(int((y1 + y2) / 2))
        elif abs(x1 - x2) <= 2 and abs(y2 - y1) > screenshot.shape[0] * 0.3:
            xs.append(int((x1 + x2) / 2))

    if not xs or not ys:
        h, w = screenshot.shape[:2]
        for candidate in (8, 6, 10, 12):
            if min(h, w) % candidate == 0:
                cell = min(h, w) // candidate
                return candidate, candidate, cell
        cell = min(h, w) // 8
        return 8, 8, cell

    xs = sorted(set(xs))
    ys = sorted(set(ys))
    # Use the median spacing between adjacent detected lines.
    dxs = np.diff(xs)
    dys = np.diff(ys)
    cell_x = int(np.median(dxs)) if len(dxs) else screenshot.shape[1] // 8
    cell_y = int(np.median(dys)) if len(dys) else screenshot.shape[0] // 8
    cell = max(8, int(np.mean([cell_x, cell_y])))
    rows = max(2, screenshot.shape[0] // cell)
    cols = max(2, screenshot.shape[1] // cell)
    return rows, cols, cell


def _classify_cell(cell_img: np.ndarray) -> Tuple[int, Optional[int]]:
    """
    Classify a single cell image.

    Returns
    -------
    (cell_type, number_or_none)

    Heuristics
    ----------
    * WALL cells are predominantly dark (low mean brightness) and have no
      significant text region.
    * START cells usually contain a coloured circle (we detect a saturated
      disc via HSV thresholding).
    * NUMBERED cells contain a digit. We OCR-lite the digit using template
      matching against pre-rendered digit templates.
    * EMPTY cells have a uniform light background.
    """
    if cell_img is None or cell_img.size == 0:
        return int(CellType.EMPTY), None

    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    h, w = gray.shape

    # WALL detection: dark cell with low variance.
    if mean_brightness < 80:
        return int(CellType.WALL), None

    # Colour analysis for START / GOAL detection.
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    # Saturated pixels = coloured markers.
    sat_mask = cv2.inRange(hsv, (0, 80, 80), (180, 255, 255))
    coloured_ratio = float(np.count_nonzero(sat_mask) / (h * w))

    # Digit detection via MSER / contour area.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 8,
    )
    # Remove tiny noise.
    n_labels, _ = cv2.connectedComponents(binary)
    contour_area = float(np.count_nonzero(binary) / (h * w))

    # START cell: usually a saturated disc, often blue/green.
    if coloured_ratio > 0.18:
        # Distinguish START from GOAL by hue: Zip uses a greenish-blue for
        # the start and a brighter green for the goal.
        mean_hue = float(np.mean(hsv[:, :, 0][sat_mask > 0])) if np.any(sat_mask) else 0
        if 40 <= mean_hue <= 90:
            return int(CellType.GOAL), None
        return int(CellType.START), None

    # If the cell has a sizeable dark blob and is bright overall, it likely
    # contains a digit -> NUMBERED.
    if 0.05 < contour_area < 0.55 and mean_brightness > 100:
        number = _recognise_digit(cell_img)
        if number is not None:
            return int(CellType.NUMBERED), number
        # Fallback: treat as NUMBERED with unknown id (the caller will
        # re-number by reading order).
        return int(CellType.NUMBERED), None

    return int(CellType.EMPTY), None


# Pre-rendered digit templates (lazy-built).
_DIGIT_TEMPLATES: Optional[dict] = None


def _build_digit_templates() -> dict:
    """Render 1-40 templates with OpenCV so we can template-match against them."""
    templates: dict = {}
    font = cv2.FONT_HERSHEY_SIMPLEX
    for d in range(1, 41):
        img = np.full((40, 40), 255, dtype=np.uint8)
        text = str(d)
        text_size, _ = cv2.getTextSize(text, font, 0.8, 2)
        text_x = (40 - text_size[0]) // 2
        text_y = (40 + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), font, 0.8, 0, 2, cv2.LINE_AA)
        templates[d] = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2,
        )
    return templates


def _recognise_digit(cell_img: np.ndarray) -> Optional[int]:
    """
    Template-match the cell against pre-rendered digit templates.

    Returns the digit (1-9) with the highest normalised correlation, or
    ``None`` if no template exceeds the confidence threshold.
    """
    global _DIGIT_TEMPLATES
    if _DIGIT_TEMPLATES is None:
        _DIGIT_TEMPLATES = _build_digit_templates()

    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 8,
    )
    binary = cv2.resize(binary, (40, 40))

    best_digit: Optional[int] = None
    best_score: float = 0.0
    for digit, tmpl in _DIGIT_TEMPLATES.items():
        res = cv2.matchTemplate(binary, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_digit = digit
    if best_score >= 0.45:
        return best_digit
    return None


def parse_canvas(
    screenshot: np.ndarray,
    canvas_bbox: Tuple[int, int, int, int],
    dom_state: dict = None,
) -> GameState:
    """
    Run the full OpenCV pipeline over a canvas screenshot and return a
    ``GameState`` ready for the solver.
    """
    if dom_state and dom_state.get('count', 0) > 0:
        import math
        cols = int(math.sqrt(dom_state['count']))
        rows = cols
        cell_size = int(canvas_bbox[2] / cols)
    else:
        try:
            rows, cols, cell_size = _detect_grid_shape(screenshot)
        except Exception as exc:
            raise VisionError(f"Grid detection failed: {exc}") from exc

    grid = np.zeros((rows, cols), dtype=np.int32)
    cells: List[Cell] = []
    start: Optional[Tuple[int, int]] = None
    goal: Optional[Tuple[int, int]] = None
    numbered_cells: List[Cell] = []

    # Slice the screenshot into cells. Use the smaller of cell_size vs the
    # available canvas so we never read past the buffer.
    # Find the active cell color by sampling the top-left 5x5 pixels
    # of the first known numbered cell.
    active_bgr = None
    if dom_state and dom_state.get('mapping'):
        first_idx = int(next(iter(dom_state['mapping'].keys())))
        first_r, first_c = first_idx // cols, first_idx % cols
        y0, x0 = first_r * cell_size, first_c * cell_size
        patch = screenshot[y0+5:y0+10, x0+5:x0+10]
        if patch.size > 0:
            active_bgr = np.mean(patch, axis=(0,1))

    h_canvas, w_canvas = screenshot.shape[:2]
    for r in range(rows):
        for c in range(cols):
            y0 = r * cell_size
            x0 = c * cell_size
            y1 = min(y0 + cell_size, h_canvas)
            x1 = min(x0 + cell_size, w_canvas)
            if y0 >= y1 or x0 >= x1:
                continue
            cell_img = screenshot[y0:y1, x0:x1]
            if dom_state:
                idx = r * cols + c
                mapping = {int(k): v for k, v in dom_state.get('mapping', {}).items()}
                if idx in mapping:
                    number = mapping[idx]
                    cell_type = int(CellType.NUMBERED)
                else:
                    number = None
                    cell_type = int(CellType.EMPTY)
                    if active_bgr is not None:
                        # Sample the top-left to avoid any path lines drawn through the center
                        patch = cell_img[5:10, 5:10]
                        if patch.size > 0:
                            mean_color = np.mean(patch, axis=(0,1))
                            dist = np.linalg.norm(mean_color - active_bgr)
                            
                            # If it's significantly different from the active white color,
                            # it is an invisible beige wall cutout.
                            if dist > 10.0:
                                cell_type = int(CellType.WALL)
            else:
                try:
                    cell_type, number = _classify_cell(cell_img)
                except Exception:
                    cell_type, number = int(CellType.EMPTY), None

            grid[r, c] = cell_type
            cell = Cell(
                row=r, col=c, cell_type=cell_type, number=number,
                pixel_center=(x0 + cell_size // 2, y0 + cell_size // 2),
            )
            cells.append(cell)
            if cell_type == int(CellType.START):
                start = (r, c)
            elif cell_type == int(CellType.GOAL):
                goal = (r, c)
            elif cell_type == int(CellType.NUMBERED):
                numbered_cells.append(cell)

    # Re-assign numbers to numbered cells in row-major reading order if any
    # cell returned None (i.e. OCR failed). This keeps the solver strict
    # about ordering even with imperfect vision.
    numbered_cells.sort(key=lambda c: (c.row, c.col))
    next_id = 1
    for cell in numbered_cells:
        if cell.number is None:
            cell.number = next_id
            next_id += 1
        else:
            next_id = max(next_id, cell.number + 1)
        grid[cell.row, cell.col] = int(CellType.NUMBERED)

    numbers = np.array(
        sorted({c.number for c in numbered_cells if c.number is not None}),
        dtype=np.int32,
    )

    if start is None and numbered_cells:
        start_c = min(numbered_cells, key=lambda c: c.number)
        start = (start_c.row, start_c.col)
        grid[start[0], start[1]] = int(CellType.START)

    if goal is None and numbered_cells:
        goal_c = max(numbered_cells, key=lambda c: c.number)
        goal = (goal_c.row, goal_c.col)
        grid[goal[0], goal[1]] = int(CellType.GOAL)
    if start is None or goal is None:
        raise VisionError(
            "Could not identify both a START and a GOAL cell on the board."
        )

    return GameState(
        grid=grid,
        numbers=numbers,
        start=start,
        goal=goal,
        cells=cells,
        canvas_bbox=canvas_bbox,
        cell_size=cell_size,
        screenshot=screenshot,
    )


# ---------------------------------------------------------------------------
# Top-level async entry point
# ---------------------------------------------------------------------------

async def capture_state(
    headless: bool = False,
    game_url: str = DEFAULT_GAME_URL,
) -> Tuple[object, GameState]:
    """
    Convenience coroutine: launch a browser, capture the canvas and return a
    fully populated ``GameState`` together with the live Playwright ``page``
    (so the executor can reuse it to drag through the solution).
    """
    pw, context = await launch_browser(headless=headless)
    try:
        page, screenshot, bbox = await capture_canvas_screenshot(
            context, game_url=game_url
        )
        dom_state = await page.evaluate("""() => {
            let cells = document.querySelectorAll("[data-cell-idx]");
            let mapping = {};
            cells.forEach(c => {
                let idx = parseInt(c.getAttribute("data-cell-idx"), 10);
                let content = c.querySelector("[data-cell-content='true']");
                if (content && content.getAttribute("aria-label")) {
                    let num = parseInt(content.getAttribute("aria-label"), 10);
                    if (!isNaN(num)) mapping[idx] = num;
                }
            });
            let bg = window.getComputedStyle(document.body).backgroundColor;
            return { mapping: mapping, count: cells.length, bg: bg };
        }""")
        state = parse_canvas(screenshot, bbox, dom_state=dom_state)
        return page, state
    except VisionError:
        await pw.stop()
        raise
    except Exception as exc:
        await pw.stop()
        raise VisionError(f"Unexpected error during state capture: {exc}") from exc


# ---------------------------------------------------------------------------
# Module CLI - lets you debug the vision pipeline in isolation.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def _debug() -> None:
        page, state = await capture_state(headless=False)
        print("Grid shape :", state.shape)
        print("Start      :", state.start)
        print("Goal       :", state.goal)
        print("Numbers    :", state.numbers)
        print(state.grid)
        print("Keeping browser open for 30s for inspection...")
        await asyncio.sleep(30)

    asyncio.run(_debug())
