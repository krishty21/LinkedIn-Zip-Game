# LinkedIn Zip Solver (Optimized v2)

A hybrid ML / constraint-programming pipeline that solves the LinkedIn
**Zip** puzzle game via browser automation.

## What's new in v2

* **DOM-first vision**: reads cell data directly from the page DOM
  (`[data-cell-idx]`, `aria-label`) for pixel-perfect accuracy — no OCR.
* **Color-based wall detection**: compares each cell's background against
  the active (white) cell colour to detect invisible beige walls.
* **Edge Wall Detection**: Extracts internal black walls using OpenCV by scanning pixel gaps between cells on the raw screenshot, preventing the solver from clipping through internal obstacles.
* **Optimized solver**: Warnsdorff-style heuristic move ordering + deadlock detection + connectedness pruning + **strict mathematical edge routing**.
* **Trained CNN**: ships with `zip_cnn.pth` weights trained on 500+ synthetic boards. Used as a move-ordering prior; falls back to plain DFS if torch is unavailable.
* **Blazing Fast Executor**: Configured for near-instantaneous drawing of straight lines (sub-millisecond delay) with zero jitter.
* **Robust types**: the `'str' // int` crash from v1 is eliminated via universal type coercion.

## Quick start

```bash
# 2. Train the CNN (optional but recommended — ~2 min on CPU)
./venv/bin/python train.py --num-games 500 --epochs 25

# 3. Run the full pipeline
./venv/bin/python main.py --skip-setup

# Headless solve (requires prior login):
./venv/bin/python main.py --skip-setup --headless --solve-only

```

## Architecture

### Vision (`vision.py`)

1. **Primary**: DOM extraction via `page.evaluate()`. Queries every
   `[data-cell-idx]` element, reads `aria-label` for waypoint numbers,
   and `getComputedStyle().backgroundColor` for wall detection (using global mode).
2. **Edge Walls**: Crops the grid from a full-page OpenCV screenshot and scans the exact midpoints between all adjacent cells to build a strict `h_walls` and `v_walls` matrix.
3. **Fallback**: OpenCV adaptive-threshold + Hough-line grid detection
   on a screenshot, used only if DOM extraction fails.

### Solver (`solver.py` + `model.py`)

* **DFS backbone** with strict pruning:
  * Waypoint ordering (must visit numbered cells in order).
  * Connectedness (unvisited cells must stay 4-connected).
  * Goal timing (goal cell only entered when all cells visited).
  * Deadlock detection (isolated unvisited cells).
* **Move ordering** (best-first):
  1. CNN policy prior (if `zip_cnn.pth` exists).
  2. Warnsdorff heuristic (prefer neighbours with fewer onward moves).
  3. Waypoint attraction (prefer moves towards the next waypoint).

### Executor (`executor.py`)

* Cubic Bezier interpolation between cell centres.
* Adaptive step count (straight = fewer, corners = more).
* Tapered jitter (zero at cell centres so endpoints are exact).
* Retry on failure with slower, smoother parameters.

### Training (`train.py`)

* Generates random solvable boards (5×5 to 8×8).
* Uses the true solution path as supervision.
* AlphaZero-style composite loss: policy cross-entropy + value MSE.

## CLI flags (`main.py`)

| Flag | Effect |
|------|--------|
| `--skip-setup` | Skip Phase 1 |
| `--solve-only` | Skip packaging |
| `--package-only` | Rebuild zip only |
| `--headless` | Headless browser |
| `--no-cnn` | Plain DFS (no CNN) |
| `--train` | Train CNN before solving |
| `--max-iterations` | DFS node budget (default 5,000,000) |

## Notebook

Open `setup_notebook.ipynb` in Jupyter to:
1. Verify the environment.
2. Run synthetic solver tests (5 boards).
3. Train the CNN.
4. Launch the live solver against LinkedIn.

## File layout

```
linkedin_zip_solver/
├── vision.py          # DOM-first grid parser
├── solver.py          # DFS + CNN + pruning
├── model.py           # PyTorch CNN (graceful fallback)
├── executor.py        # Bezier mouse drag
├── train.py           # Synthetic self-play trainer
├── main.py            # orchestrator
├── requirements.txt   # dependency pin set
├── setup_notebook.ipynb
├── zip_cnn.pth        # trained weights (optional)
└── README.md
```
