"""
main.py — Master orchestration (Optimized v2)
==============================================

Chains: vision → solver → executor.

CLI:
    --headless        Headless browser
    --no-cnn          Plain DFS (no CNN prior)
    --train           Train CNN before solving
    --max-iterations  DFS node budget
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT: Path = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from executor import ExecutorConfig, execute_path
from solver import SolverConfig, solve as solve_state
from vision import VisionError, capture_state


def run_training() -> bool:
    print("\n=== Training CNN model ===")
    script = PROJECT_ROOT / "train.py"
    if not script.exists():
        print(f"[main] train.py not found")
        return False
    try:
        r = subprocess.run(
            [sys.executable, str(script), "--num-games", "300", "--epochs", "20"],
            cwd=str(PROJECT_ROOT),
        )
        return r.returncode == 0
    except OSError as exc:
        print(f"[main] Training failed: {exc}")
        return False


async def run_vision(headless: bool):
    print("\n=== Vision & state capture ===")
    try:
        page, state = await capture_state(headless=headless)
    except VisionError as exc:
        print(f"[main] Vision error: {exc}")
        return None, None
    except Exception:
        import traceback
        print("[main] Unexpected vision error:")
        traceback.print_exc()
        return None, None

    print(f"[main] Grid: {state.shape[0]}x{state.shape[1]} "
          f"(cell_size={state.cell_size}px)")
    print(f"[main] Start: {state.start}   Goal: {state.goal}")
    print(f"[main] Waypoints: {state.numbers.tolist()}")
    return page, state


def run_solver(state, max_iterations: int, use_cnn: bool):
    print("\n=== Hybrid DFS + CNN solver ===")
    config = SolverConfig(
        max_iterations=max_iterations,
        use_cnn_prior=use_cnn,
        verbose=True,
    )
    path = solve_state(state, config=config)
    if path is None:
        print("[main] No solution found within iteration budget.")
        return None
    print(f"[main] Solution found: {len(path)} cells")
    return path


async def run_executor(page, state, path) -> bool:
    print("\n=== Autonomous input execution ===")
    if page is None:
        print("[main] No live page — skipping execution.")
        return False
    cfg = ExecutorConfig()
    ok = await execute_path(page, path, state, config=cfg)
    print(f"[main] Execution {'succeeded' if ok else 'failed'}.")
    await asyncio.sleep(0.5)
    return ok


async def amain(args: argparse.Namespace) -> int:
    if args.train:
        run_training()

    page, state = await run_vision(headless=args.headless)
    if state is None:
        print("[main] Vision failed — aborting.")
        return 1

    path = run_solver(state,
                      max_iterations=args.max_iterations,
                      use_cnn=not args.no_cnn)
    if path is None:
        return 1

    await run_executor(page, state, path)
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LinkedIn Zip solver.")
    p.add_argument("--skip-setup", action="store_true",
                   help="(Legacy, kept for compatibility.)")
    p.add_argument("--solve-only", action="store_true",
                   help="(Legacy, kept for compatibility.)")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--no-cnn", action="store_true")
    p.add_argument("--train", action="store_true",
                   help="Train CNN before solving.")
    p.add_argument("--max-iterations", type=int, default=5_000_000)
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\n[main] Interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
