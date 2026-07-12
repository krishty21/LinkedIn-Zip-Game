import random
import torch
import numpy as np
from pathlib import Path
from vision import Cell, GameState, CellType
from solver import ZipSolver, SolverConfig
from model import train_on_samples, DEFAULT_WEIGHTS_PATH

def generate_random_board(rows=8, cols=8, min_length=15, max_length=40):
    for _ in range(1000):
        visited = np.zeros((rows, cols), dtype=bool)
        start_r, start_c = random.randint(0, rows-1), random.randint(0, cols-1)
        path = [(start_r, start_c)]
        visited[start_r, start_c] = True
        
        target_len = random.randint(min_length, max_length)
        
        while len(path) < target_len:
            r, c = path[-1]
            neighbors = []
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                    neighbors.append((nr, nc))
            
            if not neighbors:
                break
                
            nr, nc = random.choice(neighbors)
            path.append((nr, nc))
            visited[nr, nc] = True
            
        if len(path) >= min_length:
            num_waypoints = max(2, len(path) // 5)
            wp_indices = [0, len(path)-1]
            
            interior_cands = list(range(3, len(path)-3, 3))
            if interior_cands:
                k = min(len(interior_cands), num_waypoints - 2)
                wp_indices.extend(random.sample(interior_cands, k))
                
            wp_indices = sorted(list(set(wp_indices)))
            
            grid = np.zeros((rows, cols), dtype=np.int32)
            cells = []
            for r in range(rows):
                for c in range(cols):
                    if (r, c) in path:
                        idx = path.index((r, c))
                        if idx in wp_indices:
                            wp_num = wp_indices.index(idx) + 1
                            if wp_num == 1:
                                cell_type = int(CellType.START)
                                num = wp_num
                            elif wp_num == len(wp_indices):
                                cell_type = int(CellType.GOAL)
                                num = wp_num
                            else:
                                cell_type = int(CellType.NUMBERED)
                                num = wp_num
                        else:
                            cell_type = int(CellType.EMPTY)
                            num = None
                    else:
                        cell_type = int(CellType.WALL)
                        num = None
                        
                    cells.append(Cell(row=r, col=c, cell_type=cell_type, number=num))
                    grid[r, c] = cell_type
                    
            
            numbers = []
            for cell in cells:
                if cell.cell_type == int(CellType.NUMBERED) and cell.number is not None:
                    numbers.append(cell.number)
            numbers = np.array(sorted(numbers), dtype=np.int32)
            
            start_cell = next(c for c in cells if c.cell_type == int(CellType.START))
            goal_cell = next(c for c in cells if c.cell_type == int(CellType.GOAL))
            
            state = GameState(
                grid=grid,
                numbers=numbers,
                start=(start_cell.row, start_cell.col),
                goal=(goal_cell.row, goal_cell.col),
                cells=cells
            )
            return state, path
            
    raise RuntimeError('Failed to generate board')

def generate_samples(num_games=100):
    samples = []
    config = SolverConfig(max_iterations=100000, use_cnn_prior=False, verbose=False)
    
    games_generated = 0
    while games_generated < num_games:
        try:
            state, true_path = generate_random_board(rows=8, cols=8, min_length=20, max_length=45)
            
            solver = ZipSolver(state, model=None, config=config)
            
            # Since we know the true path, we can directly generate training samples
            # without running the DFS solver!
            
            visited = np.zeros((8, 8), dtype=bool)
            next_wp_idx = 1 # start at waypoint 1 (we are already at waypoint 0)
            
            for i in range(len(true_path) - 1):
                r, c = true_path[i]
                visited[r, c] = True
                
                # Check if we advanced a waypoint on this step
                if solver.grid[r, c] == int(CellType.NUMBERED) or solver.grid[r, c] == int(CellType.START):
                    if next_wp_idx < len(solver.waypoints) and solver.waypoints[next_wp_idx-1] == (r, c):
                        # No, wait. The current cell is a waypoint.
                        # Actually _advance_waypoint advances if we just stepped ON it.
                        pass # handled below
                        
                # Just use solver._advance_waypoint
                next_wp_idx = solver._advance_waypoint(r, c, next_wp_idx)
                
                # Build tensor
                state_tensor = solver._build_input_tensor(r, c, visited, next_wp_idx)
                
                # True next move
                nr, nc = true_path[i+1]
                dr, dc = nr - r, nc - c
                
                # MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
                action_idx = -1
                if (dr, dc) == (-1, 0): action_idx = 0
                elif (dr, dc) == (1, 0): action_idx = 1
                elif (dr, dc) == (0, -1): action_idx = 2
                elif (dr, dc) == (0, 1): action_idx = 3
                
                if action_idx == -1:
                    raise ValueError(f"Invalid move in path: {(dr,dc)}")
                    
                policy = torch.zeros(4, dtype=torch.float32)
                policy[action_idx] = 1.0
                
                value = 1.0
                
                samples.append((state_tensor.squeeze(0), policy, value))
                
            games_generated += 1
            if games_generated % 50 == 0:
                print(f"Generated {games_generated}/{num_games} games...")
                
        except Exception as e:
            pass
            
    return samples

if __name__ == "__main__":
    print("Generating synthetic data...")
    samples = generate_samples(num_games=1000)
    print(f"Total samples: {len(samples)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}...")
    
    # Train
    model = train_on_samples(
        samples=samples,
        epochs=15,
        lr=1e-3,
        weights_path=DEFAULT_WEIGHTS_PATH,
        device=device
    )
    
    print("Training complete! Weights saved to", DEFAULT_WEIGHTS_PATH)
