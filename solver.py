"""
solver.py
=========
Phase 3 - Hybrid DFS backtracking solver guided by the CNN policy.

Algorithm
---------
The Zip game is essentially a Hamiltonian-path puzzle with waypoint
ordering constraints. The solver explores the move tree with depth-first
search, but:

  * **Move ordering** is informed by the CNN policy: at every node, candidate
    moves are sorted by descending predicted probability so that promising
    branches are visited first. This dramatically shrinks the effective
    search space without sacrificing completeness.

  * **Strict pruning** guarantees correctness:
      1. *Waypoint ordering*: a NUMBERED cell may only be entered when its
         number equals ``next_expected_number``. Visiting a numbered cell
         out of order kills the branch immediately.
      2. *Connectedness*: after every move, the set of unvisited non-wall
         cells must remain 4-connected. If they split into multiple
         components, no Hamiltonian completion can exist - prune.
      3. *Goal timing*: the GOAL cell may only be entered once every cell
         has been visited.
      4. *Wall/revisit*: walls and already-visited cells are obviously
         illegal.

  * The solver returns a list of grid coordinates ``[(r0,c0), (r1,c1), ...]``
    representing the solved path, or ``None`` if no solution exists within
    the iteration budget.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from model import NUM_ACTIONS, ZipCNN, load_model
from vision import CellType, GameState

# ---------------------------------------------------------------------------
# Move definitions
# ---------------------------------------------------------------------------

# Action index -> (dr, dc)
MOVES: List[Tuple[int, int]] = [
    (-1, 0),  # 0 = UP
    (1, 0),   # 1 = DOWN
    (0, -1),  # 2 = LEFT
    (0, 1),   # 3 = RIGHT
]


# ---------------------------------------------------------------------------
# Search state
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SolverConfig:
    max_iterations: int = 2_000_000     # safety cap on explored nodes
    use_cnn_prior: bool = True          # disable to get plain DFS
    cnn_device: str = "cpu"
    verbose: bool = False


# ---------------------------------------------------------------------------
# Connectedness check
# ---------------------------------------------------------------------------

def _unvisited_is_connected(
    grid: np.ndarray,
    visited: np.ndarray,
    start_r: int,
    start_c: int,
) -> bool:
    """
    Return True iff the set of cells that are not walls and not visited is
    4-connected AND reachable from the current position.

    Walls are identified by ``grid == CellType.WALL``.
    """
    rows, cols = grid.shape
    unvisited_mask = (grid != int(CellType.WALL)) & (~visited)
    if not unvisited_mask.any():
        # All cells visited - this is the goal state.
        return True

    # Flood fill from the current position's neighbours only through
    # unvisited cells. If the flood-fill covers everything in
    # ``unvisited_mask``, the components are connected.
    # Use a stack-based DFS to avoid recursion limits.
    total_unvisited = int(unvisited_mask.sum())
    seen = np.zeros_like(visited, dtype=bool)

    # Start the flood fill from any unvisited neighbour of the current
    # position. If there are none, the components are disconnected (the
    # current path has walled itself in).
    stack: List[Tuple[int, int]] = []
    for dr, dc in MOVES:
        nr, nc = start_r + dr, start_c + dc
        if 0 <= nr < rows and 0 <= nc < cols and unvisited_mask[nr, nc]:
            stack.append((nr, nc))
            seen[nr, nc] = True
            break
    if not stack:
        return total_unvisited == 0

    count = 0
    while stack:
        r, c = stack.pop()
        count += 1
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and unvisited_mask[nr, nc] and not seen[nr, nc]):
                seen[nr, nc] = True
                stack.append((nr, nc))

    return count == total_unvisited


# ---------------------------------------------------------------------------
# Hybrid solver
# ---------------------------------------------------------------------------

class ZipSolver:
    """
    Hybrid DFS + CNN solver.

    Instantiate with a ``GameState`` and call ``solve()``.
    """

    def __init__(
        self,
        state: GameState,
        config: Optional[SolverConfig] = None,
        model: Optional[ZipCNN] = None,
    ) -> None:
        self.state = state
        self.config = config or SolverConfig()
        self.grid = state.grid.copy()
        self.rows, self.cols = state.grid.shape

        # Pre-compute the ordered list of numbered waypoints.
        self.waypoints: List[Tuple[int, int]] = []
        numbered = sorted(
            (c for c in state.cells
             if c.cell_type == int(CellType.NUMBERED) and c.number is not None),
            key=lambda c: c.number,
        )
        for cell in numbered:
            self.waypoints.append((cell.row, cell.col))

        # Load the CNN prior (optional).
        if self.config.use_cnn_prior and model is None:
            try:
                self.model: Optional[ZipCNN] = load_model(device=self.config.cnn_device)
            except Exception:
                self.model = None
        else:
            self.model = model

        # Stats
        self.nodes_explored: int = 0
        self.prunes_connectedness: int = 0
        self.prunes_waypoint: int = 0

    # ------------------------------------------------------------------
    # CNN move ordering
    # ------------------------------------------------------------------
    def _order_moves(
        self,
        r: int,
        c: int,
        visited: np.ndarray,
        next_waypoint_idx: int,
    ) -> List[int]:
        """
        Return the indices of legal moves, sorted best-first.

        Legality here means: in-bounds, not a wall, not visited. The
        waypoint-ordering constraint is applied *after* sorting - the
        solver simply skips illegal moves during exploration.
        """
        legal: List[int] = []
        for idx, (dr, dc) in enumerate(MOVES):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                continue
            if self.grid[nr, nc] == int(CellType.WALL):
                continue
            if visited[nr, nc]:
                continue
            legal.append(idx)

        if not legal:
            return legal

        if self.model is None:
            return legal

        # Build a state tensor and ask the model for move preferences.
        try:
            tensor = self._build_input_tensor(r, c, visited, next_waypoint_idx)
            probs, _ = self.model.predict(tensor)
            probs = probs.detach().cpu().numpy()
        except Exception:
            return legal

        # Sort by descending predicted probability.
        legal.sort(key=lambda idx: -float(probs[idx]))
        return legal

    def _build_input_tensor(
        self,
        r: int,
        c: int,
        visited: np.ndarray,
        next_waypoint_idx: int,
    ) -> torch.Tensor:
        """
        Construct the 4-channel input tensor expected by the CNN.
        """
        h, w = self.rows, self.cols
        t = np.zeros((4, h, w), dtype=np.float32)
        # Channel 0: cell type
        t[0] = self.grid.astype(np.float32)
        # Channel 1: number value (only set for the *next* waypoint so the
        # model knows which number it's chasing).
        if next_waypoint_idx < len(self.waypoints):
            wr, wc = self.waypoints[next_waypoint_idx]
            t[1, wr, wc] = 1.0
        # Channel 2: visited mask
        t[2] = visited.astype(np.float32)
        # Channel 3: current position
        t[3, r, c] = 1.0
        return torch.from_numpy(t)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(self) -> Optional[List[Tuple[int, int]]]:
        """
        Run DFS from the start cell.

        Returns the solved path as a list of ``(row, col)`` tuples ending
        at the goal, or ``None`` if no solution was found within the
        iteration budget.
        """
        start_r, start_c = self.state.start
        goal_r, goal_c = self.state.goal

        visited = np.zeros((self.rows, self.cols), dtype=bool)
        visited[start_r, start_c] = True

        # If the start cell happens to be numbered "1", count it.
        next_wp_idx = self._advance_waypoint(start_r, start_c, 0)

        path: List[Tuple[int, int]] = [(start_r, start_c)]

        if self.config.verbose:
            print(f"[solver] start={self.state.start} goal={self.state.goal} "
                  f"waypoints={self.waypoints}")

        ok = self._dfs(
            r=start_r, c=start_c,
            goal_r=goal_r, goal_c=goal_c,
            visited=visited,
            next_waypoint_idx=next_wp_idx,
            path=path,
        )

        if self.config.verbose:
            print(f"[solver] nodes_explored={self.nodes_explored} "
                  f"prunes_conn={self.prunes_connectedness} "
                  f"prunes_wp={self.prunes_waypoint}")

        if ok:
            return path
        return None

    # ------------------------------------------------------------------
    # Recursive DFS
    # ------------------------------------------------------------------
    def _dfs(
        self,
        r: int,
        c: int,
        goal_r: int,
        goal_c: int,
        visited: np.ndarray,
        next_waypoint_idx: int,
        path: List[Tuple[int, int]],
    ) -> bool:
        self.nodes_explored += 1
        if self.nodes_explored > self.config.max_iterations:
            return False

        # Goal check: every cell visited AND we're on the goal cell.
        total_cells = int((self.grid != int(CellType.WALL)).sum())
        if (r, c) == (goal_r, goal_c):
            if int(visited.sum()) == total_cells \
                    and next_waypoint_idx >= len(self.waypoints):
                return True
            # Arrived at the goal too early - the goal is a dead-end here.
            return False

        ordered_moves = self._order_moves(
            r, c, visited, next_waypoint_idx,
        )

        for idx in ordered_moves:
            dr, dc = MOVES[idx]
            nr, nc = r + dr, c + dc
            # Re-check legality (defensive - _order_moves already filters).
            if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                continue
            if self.grid[nr, nc] == int(CellType.WALL):
                continue
            if visited[nr, nc]:
                continue

            cell_type = self.grid[nr, nc]

            # --- PRUNE: waypoint ordering ---
            if cell_type == int(CellType.NUMBERED):
                expected_idx = next_waypoint_idx
                if expected_idx >= len(self.waypoints) \
                        or (nr, nc) != self.waypoints[expected_idx]:
                    self.prunes_waypoint += 1
                    continue

            # --- PRUNE: goal timing ---
            if (nr, nc) == (goal_r, goal_c):
                # Only allow stepping onto the goal when every non-wall cell
                # except the goal itself has been visited.
                total_cells = int((self.grid != int(CellType.WALL)).sum())
                if int(visited.sum()) + 1 < total_cells:
                    continue

            # Apply the move.
            visited[nr, nc] = True
            path.append((nr, nc))
            new_wp_idx = self._advance_waypoint(nr, nc, next_waypoint_idx)

            # --- PRUNE: connectedness ---
            if not _unvisited_is_connected(self.grid, visited, nr, nc):
                self.prunes_connectedness += 1
                # Undo and try the next move.
                visited[nr, nc] = False
                path.pop()
                continue

            if self._dfs(nr, nc, goal_r, goal_c, visited,
                         new_wp_idx, path):
                return True

            # Undo on failure.
            visited[nr, nc] = False
            path.pop()

        return False

    def _advance_waypoint(
        self,
        r: int,
        c: int,
        current_idx: int,
    ) -> int:
        """
        If ``(r, c)`` is the next expected waypoint, advance the index.
        Otherwise return ``current_idx`` unchanged.
        """
        if current_idx < len(self.waypoints) \
                and self.waypoints[current_idx] == (r, c):
            return current_idx + 1
        return current_idx


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def solve(state: GameState,
          config: Optional[SolverConfig] = None,
          model: Optional[ZipCNN] = None) -> Optional[List[Tuple[int, int]]]:
    """One-shot solver entry point."""
    solver = ZipSolver(state, config=config, model=model)
    return solver.solve()


# ---------------------------------------------------------------------------
# CLI - solve from a saved screenshot (debugging only)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pickle
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python solver.py <state.pkl>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        state = pickle.load(f)

    config = SolverConfig(verbose=True)
    path = solve(state, config=config)
    if path is None:
        print("No solution found.")
    else:
        print(f"Solution length: {len(path)}")
        print(path)
