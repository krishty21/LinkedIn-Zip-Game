# LinkedIn Zip Solver: Deep Learning & Autonomous Automation

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Playwright](https://img.shields.io/badge/Playwright-Chromium-green.svg)](https://playwright.dev/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An autonomous, end-to-end artificial intelligence solver for the daily **LinkedIn Zip** puzzle game. Built with a hybrid architecture combining **AlphaZero-style deep convolutional policy/value priors**, **heuristic depth-first branch-and-bound search**, and **pixel-accurate DOM browser automation**.

The repository is organized into self-contained, interactive **Jupyter Notebooks** for clean modularity, transparent experimentation, and professional demonstration.

---

## Architecture

```
linkedin_zip_solver/
├── 01_Model_Training.ipynb              # Deep CNN training & synthetic self-play
├── 02_Zip_Solver_and_Automation.ipynb   # DOM extraction, DFS solver & live Playwright play
├── requirements.txt                     # Core dependencies
├── zip_cnn.pth                          # Pre-trained CNN policy & value weights
├── .gitignore                           # Git exclusions (browser profiles, caches)
└── README.md                            # Documentation & quick start guide
```

---

## Notebook 1: Model Training (`01_Model_Training.ipynb`)

**Objective:** Generate diverse synthetic Zip puzzles and train a lightweight dual-headed policy/value neural network.

* **Synthetic Board Synthesis:** Generates constrained Hamiltonian random walks across randomized dimensions ($5 \times 5$ to $8 \times 8$), placing sequential numbered waypoints and barrier walls.
* **4-Channel Tensor Representation:**
  * Channel 0: Board wall geometry
  * Channel 1: Current agent coordinates
  * Channel 2: Spatial history of visited cells
  * Channel 3: Coordinates of the next target waypoint
* **`ZipCNN` Architecture:** 4-layer residual convolutional trunk with Batch Normalization, splitting into:
  * **Policy Head:** Logit distribution over 4 cardinal actions (UP, DOWN, LEFT, RIGHT).
  * **Value Head:** Scalar evaluation in $[-1, +1]$ predicting solution proximity.
* **AlphaZero Composite Loss:** Joint optimization of Cross-Entropy policy loss and Mean Squared Error value loss.
* **Weight Export:** Produces the optimized weights file `zip_cnn.pth` (~386 KB).

---

## Notebook 2: Solver and Automation (`02_Zip_Solver_and_Automation.ipynb`)

**Objective:** Extract the live LinkedIn Zip puzzle board from the browser DOM, compute the optimal path in milliseconds, and autonomously drag the mouse to solve it live.

* **DOM-First State Extraction:**
  * Queries `[data-cell-idx]` elements and `aria-label` tags directly inside the page via JavaScript evaluation for 100% pixel-perfect board extraction without brittle OCR.
  * Classifies barrier walls by calculating Euclidean distances in RGB computed background color space.
* **Hybrid DFS Search Engine:**
  * **Connectedness Flood-Fill:** Single-pass flood fill verifying all remaining unvisited cells stay 4-connected.
  * **Deadlock Detection:** Local $2 \times 2$ neighborhood scans that prune branches when any cell becomes trapped with 0 exits.
  * **Warnsdorff's Move Ordering:** Prioritizes moves toward cells with the fewest onward degrees, heavily mitigating dead ends.
  * **CNN Prior Bias:** Incorporates predictions from `zip_cnn.pth` to bias search order toward human-like paths.
* **Autonomous Mouse Drag Executor:**
  * Maps discrete grid coordinates to exact page viewport pixels using extracted DOM bounding rects.
  * Generates cubic Bezier curves with natural human-like velocity profiles (slow start, acceleration, deceleration at goal).
  * Executes continuous mouse drag gestures via Playwright's mouse API.
* **Game Modes:**
  * **Guest Mode (`LOGGED_IN = False`):** Navigates to LinkedIn's desktop practice URL for unlimited plays without consuming your daily attempt or requiring login.
  * **Logged-in Mode (`LOGGED_IN = True`):** Solves the official daily puzzle using your saved browser profile cookies.

---

## Quick Start

### 1. Environment Setup

Clone the repository and install the dependencies:

```bash
# 1. Install required Python packages
pip install -r requirements.txt

# 2. Install Playwright browser binaries
python -m playwright install chromium
```

### 2. Running the Notebooks

Launch JupyterLab, Jupyter Notebook, or open directly in VS Code:

```bash
jupyter notebook
```

1. **To Train / Experiment with the Model:**  
   Open and run **`01_Model_Training.ipynb`**. Cells will generate synthetic training boards, train the CNN, and save updated weights to `zip_cnn.pth`.
2. **To Solve the Live Game:**  
   Open **`02_Zip_Solver_and_Automation.ipynb`**, set your desired configuration (`LOGGED_IN = False`, `HEADLESS = False`), and run the notebook. A Chromium window will launch, capture the board, solve it in milliseconds, and execute the solution path!

---

## Benchmark Performance

| Board Size | Waypoints | Nodes Explored | Solver Latency | Path Length |
| :--- | :--- | :--- | :--- | :--- |
| $6 \times 6$ | 3 | 14 | **~10 ms** | 14 cells |
| $6 \times 6$ | 4 | 22 | **~15 ms** | 20 cells |
| $7 \times 7$ | 4 | 20 | **~19 ms** | 16 cells |
| $8 \times 8$ (Live Game) | 16–20 | 64–90 | **< 25 ms** | 64 cells |

---

## Requirements

* **Python:** 3.10+ (tested on Python 3.12 & 3.14)
* **PyTorch:** $\ge 2.1.0$
* **Playwright:** $\ge 1.40.0$
* **NumPy:** $\ge 1.24.0$
* **IPyKernel:** $\ge 6.20.0$

---

## License

MIT License. See [LICENSE](LICENSE) for details.
