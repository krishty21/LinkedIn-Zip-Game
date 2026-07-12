"""
main.py
=======
Master orchestration script.

Chains every phase of the LinkedIn Zip solver pipeline in order:

    Phase 1  setup_env.py     - provision venv + dependencies + Chromium
    Phase 2  vision.py        - capture the canvas, parse the grid tensor
    Phase 3  solver.py        - hybrid DFS + CNN solver with strict pruning
    Phase 4  executor.py      - drag the solution through Playwright
    Phase 5  package.py       - zip the workspace for redistribution

CLI flags
---------
    --skip-setup      Skip Phase 1 (useful when the venv already exists).
    --solve-only      Run Phases 2-4 and skip packaging.
    --package-only    Skip everything except Phase 5.
    --headless        Run the browser headless (requires prior login).
    --max-iterations  Override the solver node budget.
    --no-cnn          Disable the CNN prior (plain DFS).

Examples
--------
    python main.py                          # full pipeline
    python main.py --skip-setup             # venv already provisioned
    python main.py --solve-only --headless  # just solve the current game
    python main.py --package-only           # rebuild the zip
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Make sibling modules importable when run directly.
PROJECT_ROOT: Path = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from executor import ExecutorConfig, execute_path
from solver import SolverConfig, solve as solve_state
from vision import VisionError, capture_state


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_phase1_setup() -> bool:
    """Invoke setup_env.py as a subprocess so it can manage the venv."""
    print("\n=== Phase 1: Environment setup ===")
    setup_script = PROJECT_ROOT / "setup_env.py"
    if not setup_script.exists():
        print(f"[main] setup_env.py not found at {setup_script}")
        return False
    try:
        result = subprocess.run(
            [sys.executable, str(setup_script)],
            cwd=str(PROJECT_ROOT),
        )
        return result.returncode == 0
    except OSError as exc:
        print(f"[main] Phase 1 failed: {exc}")
        return False


async def run_phase2_vision(headless: bool):
    """Capture the game state. Returns (page, state) or (None, None)."""
    print("\n=== Phase 2: Vision & state capture ===")
    try:
        page, state = await capture_state(headless=headless)
    except VisionError as exc:
        print(f"[main] Vision error: {exc}")
        return None, None
    except Exception as exc:
        import traceback
        print(f"[main] Unexpected vision error:")
        traceback.print_exc()
        return None, None

    print(f"[main] Grid: {state.shape[0]}x{state.shape[1]} "
          f"(cell_size={state.cell_size}px)")
    print(f"[main] Start: {state.start}   Goal: {state.goal}")
    print(f"[main] Waypoints: {state.numbers.tolist()}")
    return page, state


def run_phase3_solver(state, max_iterations: int, use_cnn: bool):
    """Run the hybrid solver. Returns the solved path or None."""
    print("\n=== Phase 3: Hybrid DFS + CNN solver ===")
    config = SolverConfig(
        max_iterations=max_iterations,
        use_cnn_prior=use_cnn,
        verbose=True,
    )
    path = solve_state(state, config=config)
    if path is None:
        print("[main] No solution found within the iteration budget.")
        return None
    print(f"[main] Solution found: {len(path)} cells")
    return path


async def run_phase4_executor(page, state, path) -> bool:
    """Drag the solution through the canvas."""
    print("\n=== Phase 4: Autonomous input execution ===")
    if page is None:
        print("[main] No live page available - skipping execution.")
        return False
    cfg = ExecutorConfig()
    ok = await execute_path(page, path, state, config=cfg)
    if ok:
        print("[main] Solution executed successfully.")
    else:
        print("[main] Execution failed.")
    # Give the user a few seconds to see the result before the browser closes.
    await asyncio.sleep(5.0)
    return ok


def run_phase5_package() -> bool:
    """Bundle the workspace into dist/linkedin_zip_solver.zip."""
    print("\n=== Phase 5: Project packaging ===")
    package_script = PROJECT_ROOT / "package.py"
    if not package_script.exists():
        print(f"[main] package.py not found at {package_script}")
        return False
    try:
        result = subprocess.run(
            [sys.executable, str(package_script)],
            cwd=str(PROJECT_ROOT),
        )
        return result.returncode == 0
    except OSError as exc:
        print(f"[main] Phase 5 failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def amain(args: argparse.Namespace) -> int:
    # ------------------------------------------------------------------
    # Phase 1 - setup
    # ------------------------------------------------------------------
    if not args.skip_setup and not args.package_only:
        if not run_phase1_setup():
            print("[main] Phase 1 failed - aborting.")
            return 1

    if args.package_only:
        return 0 if run_phase5_package() else 1

    # ------------------------------------------------------------------
    # Phase 2 - vision
    # ------------------------------------------------------------------
    page, state = await run_phase2_vision(headless=args.headless)
    if state is None:
        print("[main] Phase 2 failed - aborting.")
        return 1

    # ------------------------------------------------------------------
    # Phase 3 - solver
    # ------------------------------------------------------------------
    path = run_phase3_solver(
        state,
        max_iterations=args.max_iterations,
        use_cnn=not args.no_cnn,
    )
    if path is None:
        return 1

    # ------------------------------------------------------------------
    # Phase 4 - executor
    # ------------------------------------------------------------------
    await run_phase4_executor(page, state, path)

    # ------------------------------------------------------------------
    # Phase 5 - packaging
    # ------------------------------------------------------------------
    if not args.solve_only:
        run_phase5_package()

    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LinkedIn Zip game autonomous solver.")
    parser.add_argument("--skip-setup", action="store_true",
                        help="Skip Phase 1 (environment setup).")
    parser.add_argument("--solve-only", action="store_true",
                        help="Run only Phases 2-4; skip packaging.")
    parser.add_argument("--package-only", action="store_true",
                        help="Run only Phase 5 (rebuild the zip).")
    parser.add_argument("--headless", action="store_true",
                        help="Run Chromium headless (requires prior login).")
    parser.add_argument("--no-cnn", action="store_true",
                        help="Disable the CNN prior - use plain DFS.")
    parser.add_argument("--max-iterations", type=int, default=2_000_000,
                        help="Maximum number of DFS nodes to explore.")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\n[main] Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
