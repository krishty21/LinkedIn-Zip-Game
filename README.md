# LinkedIn Zip Solver

A hybrid ML / constraint-programming pipeline that solves the LinkedIn
**Zip** puzzle game via browser automation.

The project is split into five phases, each implemented in its own module:

| Phase | Module           | Responsibility                                         |
|------:|------------------|--------------------------------------------------------|
|   1   | `setup_env.py`   | Create venv, install deps, fetch Chromium              |
|   2   | `vision.py`      | Playwright capture + OpenCV grid parsing               |
|   3   | `solver.py`      | Hybrid DFS + CNN solver with strict pruning            |
|   3   | `model.py`       | PyTorch CNN policy/value network                       |
|   4   | `executor.py`    | Bezier-curve mouse drag through the solved path        |
|   5   | `package.py`     | Bundle workspace into `linkedin_zip_solver.zip`        |

`main.py` orchestrates every phase sequentially.

---

## Quick start

```bash
# 1. Provision the environment (creates ./venv, installs deps + Chromium)
python setup_env.py

# 2. Run the full pipeline end-to-end
#    On first run, a visible Chromium window opens so you can log into
#    LinkedIn. Cookies are persisted in ./.browser_data/ so subsequent
#    runs can be headless.
./venv/bin/python main.py

# Or, if the venv is already provisioned:
./venv/bin/python main.py --skip-setup

# Headless solve once logged in:
./venv/bin/python main.py --skip-setup --headless --solve-only

# Just rebuild the redistributable zip:
./venv/bin/python main.py --package-only
```

---

## Architecture

### Vision (`vision.py`)

* Launches Playwright with a **persistent** browser context so LinkedIn
  login cookies survive between runs.
* Captures the canvas bounding box and screenshot.
* Runs an OpenCV pipeline:
  * Adaptive thresholding + Hough lines detect the grid shape.
  * Per-cell classification via brightness / saturation / template
    matching recovers walls, the start, the goal, and numbered
    waypoints (1, 2, 3, ...).
* Outputs a `GameState` containing the grid tensor, waypoint list,
  and cell → pixel mapping.

### Solver (`solver.py` + `model.py`)

* `ZipCNN` is a 3-conv + 2-head policy/value network (~150 KB). Given
  a 4-channel grid tensor it predicts a softmax policy over
  `{Up, Down, Left, Right}` and a scalar value in `(-1, 1)`.
* `ZipSolver` runs DFS from the start cell. At every node the legal
  moves are sorted by the CNN's predicted probability, so the most
  promising branches are explored first.
* **Strict pruning** guarantees correctness:
  * **Waypoint ordering**: numbered cells must be visited in order.
  * **Connectedness**: after every move, the unvisited non-wall cells
    must remain 4-connected (otherwise no Hamiltonian completion can
    exist - prune immediately).
  * **Goal timing**: the goal cell may only be entered once every
    non-wall cell has been visited.
* If no `.pth` weights file is present the solver silently falls back
  to plain DFS - it still works, just slower on larger boards.

### Executor (`executor.py`)

* Converts the grid path into pixel coordinates via
  `GameState.pixel_of`.
* Builds a smooth trajectory using **cubic Bezier** interpolation
  between consecutive cells, with randomised control-point sign and
  small perpendicular jitter that tapers to zero at cell centres.
* Drives the drag through Playwright's `page.mouse` API:
  `move(start)` → `down()` → walk the Bezier samples → `up()`.

### Packaging (`package.py`)

* Stages `vision.py`, `solver.py`, `model.py`, `executor.py`,
  `setup_env.py`, `package.py`, `main.py`, `requirements.txt`,
  `README.md`, and the optional `zip_cnn.pth` into `dist/`.
* Explicitly excludes `venv/`, `.venv/`, `__pycache__/`,
  `.browser_data/`, `dist/`, `.git/`, and `*.pyc` files.
* Produces `dist/linkedin_zip_solver.zip`.

---

## CLI flags (`main.py`)

| Flag                | Effect                                         |
|---------------------|------------------------------------------------|
| `--skip-setup`      | Skip Phase 1 (venv already provisioned)        |
| `--solve-only`      | Run Phases 2-4, skip packaging                 |
| `--package-only`    | Run only Phase 5 (rebuild the zip)             |
| `--headless`        | Run Chromium headless                          |
| `--no-cnn`          | Disable CNN prior, use plain DFS               |
| `--max-iterations`  | Override the DFS node budget (default 2,000,000) |

---

## Training the CNN (optional)

The solver works without trained weights (it falls back to plain DFS).
To train the network from self-play samples, collect
`(state_tensor, policy_target, value_target)` tuples and call:

```python
from model import train_on_samples
train_on_samples(samples, epochs=20, lr=1e-3)
```

The weights are saved to `zip_cnn.pth` and automatically picked up by
the solver on the next run.

---

## Error handling

Every module raises typed exceptions on failure:

* `vision.VisionError` - canvas capture / parsing failures.
* `solver.SolverConfig` validation - bad iteration budgets, etc.
* `executor` returns `False` and prints the error if the drag fails.

`main.py` surfaces the failure of each phase and aborts cleanly so you
can re-run after fixing the root cause.
