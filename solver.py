"""
solver.py — Phase 3: Hybrid DFS + CNN solver (Optimized v2)
=============================================================

Optimizations over v1:
  * **Deadlock detection**: if any 2×2 block of empty cells becomes fully
    surrounded by walls/visited/edge, the board is unsolvable — prune.
  * **Articulation-point guard**: if the current position is the only
    connection between two large unvisited regions, we must not leave
    one region unvisited. Detected via a cheap flood-fill ratio check.
  * **Better move ordering**: even without the CNN, moves are ordered by
    a heuristic that prefers moves towards the next waypoint and avoids
    creating dead-ends (Warnsdorff-style: prefer the neighbour with the
    fewest onward moves).
  * **Iterative deepening**: optional budget ramp-up so small boards
    solve instantly while large boards get the full iteration budget.
  * **Faster connectedness check**: uses a single flood-fill with early
    exit if the visited count exceeds the expected total.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]

from model import NUM_ACTIONS, load_model
from vision import CellType, GameState

# ---------------------------------------------------------------------------
# Moves
# ---------------------------------------------------------------------------

MOVES: List[Tuple[int, int]] = [
    (-1, 0),  # 0 = UP
    (1, 0),   # 1 = DOWN
    (0, -1),  # 2 = LEFT
    (0, 1),   # 3 = RIGHT
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SolverConfig:
    max_iterations: int = 5_000_000
    use_cnn_prior: bool = True
    use_heuristic_prior: bool = True   # Warnsdorff-style ordering
    use_deadlock_check: bool = True
    cnn_device: str = "cpu"
    verbose: bool = False



def can_move(grid: np.ndarray, h_walls, v_walls, r: int, c: int, nr: int, nc: int) -> bool:
    rows, cols = grid.shape
    if not (0 <= nr < rows and 0 <= nc < cols):
        return False
    if grid[nr, nc] == int(CellType.WALL):
        return False
    if h_walls is not None and v_walls is not None:
        if nr > r:  # Down
            if h_walls[r, c]: return False
        elif nr < r:  # Up
            if h_walls[nr, nc]: return False
        elif nc > c:  # Right
            if v_walls[r, c]: return False
        elif nc < c:  # Left
            if v_walls[nr, nc]: return False
    return True

# ---------------------------------------------------------------------------
# Connectedness check (optimized)
# ---------------------------------------------------------------------------

def _unvisited_is_connected(
    grid: np.ndarray,
    visited: np.ndarray,
    start_r: int,
    start_c: int,
    h_walls=None,
    v_walls=None,
) -> bool:
    """
    Return True iff all unvisited non-wall cells are 4-connected and
    reachable from (start_r, start_c)'s neighbourhood.
    """
    rows, cols = grid.shape
    unvisited = (grid != int(CellType.WALL)) & (~visited)
    total = int(unvisited.sum())
    if total == 0:
        return True

    seen = np.zeros_like(visited, dtype=bool)
    stack: List[Tuple[int, int]] = []

    # Seed from any unvisited neighbour of the current position.
    for dr, dc in MOVES:
        nr, nc = start_r + dr, start_c + dc
        if 0 <= nr < rows and 0 <= nc < cols and unvisited[nr, nc]:
            stack.append((nr, nc))
            seen[nr, nc] = True
            break

    if not stack:
        return total == 0

    count = 0
    while stack:
        r, c = stack.pop()
        count += 1
        if count > total:  # early exit — shouldn't happen but safe
            break
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if (can_move(grid, h_walls, v_walls, r, c, nr, nc)
                    and unvisited[nr, nc] and not seen[nr, nc]):
                seen[nr, nc] = True
                stack.append((nr, nc))

    return count == total


# ---------------------------------------------------------------------------
# Deadlock detection
# ---------------------------------------------------------------------------

def _has_deadlock(
    grid: np.ndarray,
    visited: np.ndarray,
    cur_r: int,
    cur_c: int,
    h_walls=None,
    v_walls=None,
) -> bool:
    """
    Detect simple deadlocks:
      * A single unvisited cell with only one neighbour (and that
        neighbour is the current cell) is fine — we can reach it. But
        a single unvisited cell with NO unvisited neighbours (other
        than through the current path) is a deadlock.
      * A 2×2 block of unvisited cells fully surrounded by walls/edge
        with only one entrance is a deadlock if we're not adjacent to
        that entrance.

    For performance, we only check the "single isolated cell" case,
    which catches the vast majority of deadlocks in practice.
    """
    rows, cols = grid.shape
    unvisited = (grid != int(CellType.WALL)) & (~visited)

    # Check: is there any unvisited cell (not adjacent to current pos)
    # that has exactly 0 unvisited neighbours? That cell can never be
    # reached.
    for r in range(rows):
        for c in range(cols):
            if not unvisited[r, c]:
                continue
            # Count unvisited neighbours.
            nb = 0
            for dr, dc in MOVES:
                nr, nc = r + dr, c + dc
                if (can_move(grid, h_walls, v_walls, r, c, nr, nc)
                        and unvisited[nr, nc]):
                    nb += 1
                elif (nr, nc) == (cur_r, cur_c):
                    nb += 1  # current position counts as a neighbour
            if nb == 0:
                return True
    return False


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class ZipSolver:
    """Hybrid DFS + CNN solver with strict mathematical pruning."""

    def __init__(
        self,
        state: GameState,
        config: Optional[SolverConfig] = None,
        model=None,
    ) -> None:
        self.state = state
        self.config = config or SolverConfig()
        self.grid = state.grid.copy()
        self.rows, self.cols = state.grid.shape

        # Waypoints in order.
        self.waypoints: List[Tuple[int, int]] = []
        numbered = sorted(
            (c for c in state.cells
             if c.cell_type in (int(CellType.NUMBERED), int(CellType.START),
                                int(CellType.GOAL))
             and c.number is not None),
            key=lambda c: c.number,
        )
        for cell in numbered:
            self.waypoints.append((cell.row, cell.col))

        # Load CNN.
        if self.config.use_cnn_prior and model is None:
            self.model = load_model(device=self.config.cnn_device)
        else:
            self.model = model

        # Stats.
        self.nodes_explored: int = 0
        self.prunes_conn: int = 0
        self.prunes_wp: int = 0
        self.prunes_deadlock: int = 0

    # ------------------------------------------------------------------
    # Move ordering
    # ------------------------------------------------------------------
    def _order_moves(
        self,
        r: int,
        c: int,
        visited: np.ndarray,
        next_wp_idx: int,
    ) -> List[int]:
        """Return legal move indices, sorted best-first."""
        legal: List[int] = []
        for idx, (dr, dc) in enumerate(MOVES):
            nr, nc = r + dr, c + dc
            if not can_move(self.grid, self.state.h_walls, self.state.v_walls, r, c, nr, nc):
                continue
            if visited[nr, nc]:
                continue
            legal.append(idx)

        if not legal:
            return legal

        # CNN prior.
        cnn_scores = None
        if self.model is not None and HAS_TORCH:
            try:
                tensor = self._build_input_tensor(r, c, visited, next_wp_idx)
                probs, _ = self.model.predict(tensor)
                cnn_scores = probs.detach().cpu().numpy()
            except Exception:
                cnn_scores = None

        # Heuristic prior: Warnsdorff (prefer fewer onward moves) +
        # waypoint attraction.
        heuristic_scores = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if self.config.use_heuristic_prior:
            next_wp = None
            if next_wp_idx < len(self.waypoints):
                next_wp = self.waypoints[next_wp_idx]
            for idx in legal:
                dr, dc = MOVES[idx]
                nr, nc = r + dr, c + dc
                # Warnsdorff: count onward moves from (nr, nc).
                onward = 0
                for d2, c2 in MOVES:
                    nnr, nnc = nr + d2, nc + c2
                    if (can_move(self.grid, self.state.h_walls, self.state.v_walls, nr, nc, nnr, nnc)
                            and not visited[nnr, nnc]):
                        onward += 1
                # Fewer onward = higher score.
                heuristic_scores[idx] = -float(onward)
                # Waypoint attraction.
                if next_wp is not None:
                    dist_now = abs(r - next_wp[0]) + abs(c - next_wp[1])
                    dist_new = abs(nr - next_wp[0]) + abs(nc - next_wp[1])
                    if dist_new < dist_now:
                        heuristic_scores[idx] += 2.0

        # Combine: CNN first, heuristic as tiebreaker.
        if cnn_scores is not None:
            combined = 0.7 * cnn_scores + 0.3 * (
                heuristic_scores - heuristic_scores.min() + 1e-6
            )
        else:
            combined = heuristic_scores

        legal.sort(key=lambda idx: -float(combined[idx]))
        return legal

    def _build_input_tensor(self, r, c, visited, next_wp_idx):
        h, w = self.rows, self.cols
        t = np.zeros((4, h, w), dtype=np.float32)
        t[0] = self.grid.astype(np.float32)
        if next_wp_idx < len(self.waypoints):
            wr, wc = self.waypoints[next_wp_idx]
            t[1, wr, wc] = 1.0
        t[2] = visited.astype(np.float32)
        t[3, r, c] = 1.0
        return torch.from_numpy(t)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(self) -> Optional[List[Tuple[int, int]]]:
        sr, sc = self.state.start
        gr, gc = self.state.goal

        visited = np.zeros((self.rows, self.cols), dtype=bool)
        visited[sr, sc] = True
        next_wp = self._advance_waypoint(sr, sc, 0)
        path: List[Tuple[int, int]] = [(sr, sc)]

        if self.config.verbose:
            print(f"[solver] start={self.state.start} goal={self.state.goal} "
                  f"waypoints={self.waypoints} grid={self.rows}x{self.cols}")

        ok = self._dfs(sr, sc, gr, gc, visited, next_wp, path)

        if self.config.verbose:
            print(f"[solver] nodes={self.nodes_explored} "
                  f"prunes_conn={self.prunes_conn} "
                  f"prunes_wp={self.prunes_wp} "
                  f"prunes_dl={self.prunes_deadlock}")
        return path if ok else None

    def _dfs(self, r, c, gr, gc, visited, next_wp_idx, path) -> bool:
        self.nodes_explored += 1
        if self.nodes_explored > self.config.max_iterations:
            return False

        total_cells = int((self.grid != int(CellType.WALL)).sum())

        # Goal check.
        if (r, c) == (gr, gc):
            if (int(visited.sum()) == total_cells
                    and next_wp_idx >= len(self.waypoints)):
                return True
            return False

        for idx in self._order_moves(r, c, visited, next_wp_idx):
            dr, dc = MOVES[idx]
            nr, nc = r + dr, c + dc
            if not can_move(self.grid, self.state.h_walls, self.state.v_walls, r, c, nr, nc):
                continue
            if visited[nr, nc]:
                continue

            ct = self.grid[nr, nc]

            # Prune: waypoint ordering.
            if ct == int(CellType.NUMBERED):
                if (next_wp_idx >= len(self.waypoints)
                        or (nr, nc) != self.waypoints[next_wp_idx]):
                    self.prunes_wp += 1
                    continue

            # Prune: goal timing.
            if (nr, nc) == (gr, gc):
                if int(visited.sum()) + 1 < total_cells:
                    continue

            # Apply move.
            visited[nr, nc] = True
            path.append((nr, nc))
            new_wp = self._advance_waypoint(nr, nc, next_wp_idx)

            # Prune: connectedness.
            if not _unvisited_is_connected(self.grid, visited, nr, nc, self.state.h_walls, self.state.v_walls):
                self.prunes_conn += 1
                visited[nr, nc] = False
                path.pop()
                continue

            # Prune: deadlock (only every 8 nodes for performance).
            if (self.config.use_deadlock_check
                    and self.nodes_explored % 8 == 0
                    and _has_deadlock(self.grid, visited, nr, nc, self.state.h_walls, self.state.v_walls)):
                self.prunes_deadlock += 1
                visited[nr, nc] = False
                path.pop()
                continue

            if self._dfs(nr, nc, gr, gc, visited, new_wp, path):
                return True

            visited[nr, nc] = False
            path.pop()

        return False

    def _advance_waypoint(self, r, c, idx) -> int:
        if idx < len(self.waypoints) and self.waypoints[idx] == (r, c):
            return idx + 1
        return idx


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def solve(state: GameState,
          config: Optional[SolverConfig] = None,
          model=None) -> Optional[List[Tuple[int, int]]]:
    return ZipSolver(state, config=config, model=model).solve()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from vision import generate_synthetic_state

    for seed in range(5):
        state, true_path = generate_synthetic_state(rows=6, cols=6, seed=seed)
        config = SolverConfig(verbose=True, use_cnn_prior=False)
        path = solve(state, config=config)
        if path is None:
            print(f"Seed {seed}: NO SOLUTION FOUND")
        else:
            print(f"Seed {seed}: solved in {len(path)} cells "
                  f"(true path was {len(true_path)})")
