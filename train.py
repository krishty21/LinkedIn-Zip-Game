"""
train.py — Self-play training for the ZipCNN policy/value network.
====================================================================

Generates diverse synthetic Zip boards, solves them with plain DFS,
and trains the CNN to predict:
  * policy: the action taken at each state along the solution path.
  * value: +1 for states on the winning path, -1 for dead-end states.

Usage:
    python train.py [--num-games 500] [--epochs 30] [--board-size 6]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]

from vision import Cell, CellType, GameState, generate_synthetic_state
from solver import ZipSolver, SolverConfig
from model import train_on_samples, DEFAULT_WEIGHTS_PATH, load_model, NUM_ACTIONS

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _move_to_action(dr: int, dc: int) -> int:
    if (dr, dc) == (-1, 0): return 0
    if (dr, dc) == (1, 0):  return 1
    if (dr, dc) == (0, -1): return 2
    if (dr, dc) == (0, 1):  return 3
    raise ValueError(f"Invalid move: ({dr}, {dc})")


def generate_samples(num_games: int = 500,
                     board_sizes: List[int] = [5, 6, 7, 8],
                     verbose: bool = True) -> list:
    """
    Generate training samples from solved synthetic boards.

    Each sample is (state_tensor (4,H,W), policy (4,), value float).
    """
    if not HAS_TORCH:
        if verbose:
            print("torch not available — cannot generate tensor samples.")
        return []

    samples = []
    config = SolverConfig(
        max_iterations=500_000,
        use_cnn_prior=False,
        use_heuristic_prior=True,
        use_deadlock_check=True,
        verbose=False,
    )

    for i in range(num_games):
        size = random.choice(board_sizes)
        n_wp = random.choice([2, 3, 4, 5])

        try:
            state, true_path = generate_synthetic_state(
                rows=size, cols=size, num_waypoints=n_wp, seed=i
            )
        except Exception:
            continue

        # We already have the true path — build samples directly.
        solver = ZipSolver(state, model=None, config=config)
        visited = np.zeros((size, size), dtype=bool)
        next_wp = solver._advance_waypoint(state.start[0], state.start[1], 0)

        for step_idx in range(len(true_path) - 1):
            r, c = true_path[step_idx]
            visited[r, c] = True
            next_wp = solver._advance_waypoint(r, c, next_wp)

            state_tensor = solver._build_input_tensor(r, c, visited, next_wp)

            nr, nc = true_path[step_idx + 1]
            dr, dc = nr - r, nc - c
            try:
                action = _move_to_action(dr, dc)
            except ValueError:
                continue

            policy = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
            policy[action] = 1.0

            # Value: +1 for states on the winning path.
            # We also add a slight decay so early states have slightly
            # lower value than later states (closer to goal = higher value).
            progress = step_idx / max(1, len(true_path) - 1)
            value = -1.0 + 2.0 * progress  # ranges from -1 to +1

            samples.append((
                state_tensor.squeeze(0).clone(),
                policy,
                value,
            ))

        if verbose and (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_games} games, "
                  f"{len(samples)} samples total")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Train the ZipCNN model.")
    parser.add_argument("--num-games", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, default=str(DEFAULT_WEIGHTS_PATH))
    args = parser.parse_args()

    if not HAS_TORCH:
        print("torch is not installed. Install it first:")
        print("  pip install torch")
        return 1

    print(f"Generating {args.num_games} synthetic games...")
    samples = generate_samples(num_games=args.num_games, verbose=True)
    print(f"Total samples: {len(samples)}")

    if not samples:
        print("No samples generated — nothing to train.")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device} for {args.epochs} epochs...")

    model = train_on_samples(
        samples=samples,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weights_path=Path(args.output),
        device=device,
    )

    if model is not None:
        print(f"Training complete! Weights saved to {args.output}")
        print(f"  File size: {Path(args.output).stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
