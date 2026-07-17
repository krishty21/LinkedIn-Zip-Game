"""
main.py — Master orchestration (Optimized v2)
==============================================

Chains: vision → solver → executor.

CLI:
    --headless        Headless browser
    --guest           Use a temporary browser profile (not logged in, unlimited plays)
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

# Pre-import model to ensure PyTorch's heavy DLL loading happens
# BEFORE we open the browser and start the in-game timer.
import model
from executor import ExecutorConfig, execute_path
from solver import SolverConfig, solve as solve_state
from vision import VisionError, capture_state, GUEST_GAME_URL


def run_training() -> bool:
    print("\nTraining CNN model...")
    script = PROJECT_ROOT / "train.py"
    if not script.exists():
        print("Oops, I couldn't find train.py.")
        return False
    try:
        r = subprocess.run(
            [sys.executable, str(script), "--num-games", "300", "--epochs", "20"],
            cwd=str(PROJECT_ROOT),
        )
        return r.returncode == 0
    except OSError as exc:
        print(f"Training failed: {exc}")
        return False


async def run_vision(headless: bool, guest: bool = False):
    print("\nStarting vision and state capture...")
    context_dir = None
    game_url = None  # use default (linkedin.com/games/zip)
    if guest:
        import tempfile
        context_dir = Path(tempfile.mkdtemp(prefix="zip_guest_"))
        game_url = GUEST_GAME_URL  # direct URL bypassing iframe
        print("Opening up a fresh browser in guest mode.")
    try:
        page, state = await capture_state(headless=headless, context_dir=context_dir,
                                          **({"game_url": game_url} if game_url else {}))
    except VisionError as exc:
        print(f"Trouble reading the board: {exc}")
        return None, None
    except Exception:
        import traceback
        print("Something unexpected went wrong while reading the board:")
        traceback.print_exc()
        return None, None

    print(f"Found a {state.shape[0]}x{state.shape[1]} puzzle board!")
    print(f"Starting at {state.start} and aiming for {state.goal}.")
    print(f"We need to hit these waypoints: {state.numbers.tolist()}")
    return page, state


def run_solver(state, max_iterations: int, use_cnn: bool):
    print("\nRunning Hybrid DFS + Heuristic solver...")
    config = SolverConfig(
        max_iterations=max_iterations,
        use_cnn_prior=use_cnn,
        verbose=True,
    )
    path = solve_state(state, config=config)
    if path is None:
        print("Couldn't find a solution in time.")
        return None
    print(f"Found the answer! The path is {len(path)} cells long.")
    return path


async def run_executor(page, state, path) -> bool:
    print("\nBeginning autonomous input execution...")
    if page is None:
        print("There is no live browser page, so I can't draw the path.")
        return False
    cfg = ExecutorConfig()
    ok = await execute_path(page, path, state, config=cfg)
    print(f"Execution {'was successful' if ok else 'failed'}.")
    await asyncio.sleep(0.5)
    return ok


async def amain(args: argparse.Namespace) -> int:
    if args.train:
        run_training()

    page, state = await run_vision(headless=args.headless, guest=args.guest)
    if state is None:
        print("Failed to read the board, stopping here.")
        return 1

    path = run_solver(state,
                      max_iterations=args.max_iterations,
                      use_cnn=args.use_cnn)
    if path is None:
        return 1

    await run_executor(page, state, path)
    
    print("\nI'll leave the browser open for an hour so you can check it out.")
    print("Feel free to close the window when you're done.")
    try:
        for _ in range(3600):
            if page.is_closed():
                break
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
        
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LinkedIn Zip solver.")
    p.add_argument("--skip-setup", action="store_true",
                   help="(Legacy, kept for compatibility.)")
    p.add_argument("--solve-only", action="store_true",
                   help="(Legacy, kept for compatibility.)")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--guest", action="store_true",
                   help="Use a temporary browser profile (not logged in).")
    p.add_argument("--use-cnn", action="store_true",
                   help="Enable CNN move prior (experimental, 10-80x slower than heuristic-only).")
    p.add_argument("--train", action="store_true",
                   help="Train CNN before solving.")
    p.add_argument("--max-iterations", type=int, default=5_000_000)
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
