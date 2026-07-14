# LinkedIn Zip Solver (Optimized v2)

A hybrid ML + constraint-programming pipeline that **autonomously solves** the
LinkedIn **Zip** puzzle game via browser automation.

## Features

* **DOM-first vision** — reads cell data directly from the page DOM
  (`[data-cell-idx]`, `aria-label`) for pixel-perfect accuracy. No OCR.
* **Color-based wall detection** — compares each cell's computed background
  against the dominant board colour to detect invisible beige walls.
* **Edge wall detection** — uses OpenCV to scan pixel midpoints between
  adjacent cells, building strict `h_walls` / `v_walls` matrices that prevent
  the solver from clipping through internal black walls.
* **Optimized DFS solver** — Warnsdorff-style heuristic move ordering +
  deadlock detection + connectedness pruning + strict edge routing.
* **Trained CNN prior** — ships with `zip_cnn.pth` (500+ synthetic boards).
  Used for move-ordering; falls back to plain DFS if PyTorch is unavailable.
* **Blazing-fast executor** — direct cell-to-cell mouse movement with
  sub-millisecond delays. No Bezier interpolation needed.
* **Guest mode** — open a fresh browser without your LinkedIn login to solve the real puzzle with unlimited practice retries.

## Quick Start

```bash
# 1. Create a virtual environment & install dependencies
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
python -m playwright install chromium

# 2. Train the CNN (optional — weights are included)
python train.py --num-games 500 --epochs 25

# 3. Run the full pipeline (opens Chrome, solves the puzzle)
python main.py --solve-only --skip-setup

# 4. Headless solve (requires prior login):
python main.py --solve-only --skip-setup --headless
```

## Notebook

Open **`setup_notebook.ipynb`** in Jupyter for an interactive experience:

1. Set `LOGGED_IN = True` to solve the **real LinkedIn puzzle** using your daily attempt (uses saved cookies).
2. Set `LOGGED_IN = False` to run in **guest mode** — opens a fresh browser, navigate to the puzzle and solve it without a login for unlimited practice.

## Architecture

### Vision (`vision.py`)

1. **DOM extraction** via `page.evaluate()` — queries every `[data-cell-idx]`
   element, reads `aria-label` for waypoint numbers, and
   `getComputedStyle().backgroundColor` for wall detection.
2. **Edge walls** — crops the grid from a full-page screenshot and scans
   midpoints between cells to build `h_walls` and `v_walls` matrices.
3. **Fallback** — OpenCV adaptive-threshold grid detection if DOM fails.

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

* Direct cell-centre-to-cell-centre mouse movement.
* Retry on failure with slower parameters.

### Training (`train.py`)

* Generates random solvable boards (5×5 to 8×8).
* Uses the true solution path as supervision.
* AlphaZero-style composite loss: policy cross-entropy + value MSE.

## CLI Flags (`main.py`)

| Flag | Effect |
|------|--------|
| `--skip-setup` | Skip environment setup phase |
| `--solve-only` | Skip packaging |
| `--headless` | Run browser headlessly |
| `--guest` | Use a temporary browser profile (unlimited plays) |
| `--no-cnn` | Plain DFS (no CNN prior) |
| `--train` | Train CNN before solving |
| `--max-iterations` | DFS node budget (default 5,000,000) |

## File Layout

```
linkedin_zip_solver/
├── vision.py            # DOM-first grid parser + edge wall detection
├── solver.py            # DFS + CNN + pruning
├── model.py             # PyTorch CNN (graceful fallback)
├── executor.py          # Fast mouse drag execution
├── train.py             # Synthetic self-play trainer
├── main.py              # CLI orchestrator
├── requirements.txt     # Dependencies
├── setup_notebook.ipynb # Interactive Jupyter notebook
├── zip_cnn.pth          # Pre-trained CNN weights
└── README.md
```

## License

MIT
