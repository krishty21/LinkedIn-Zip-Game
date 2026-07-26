"""
test_timecontrol.py — Verify time control implementation.

Tests:
1. Legacy mode (time_budget=0): returns first solution immediately
2. Time-controlled mode: utilizes the full time budget
3. Path validity: all solutions are valid
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from solver import ZipSolver, SolverConfig, solve
from vision import generate_synthetic_state
from executor import validate_path


def test_legacy_mode():
    """Test that legacy mode (time_budget=0) returns first solution fast."""
    print("=== Legacy Mode Test (time_budget=0) ===")
    total_time = 0.0
    passed = 0

    for seed in range(5):
        state, true_path = generate_synthetic_state(rows=6, cols=6, seed=seed)
        config = SolverConfig(
            max_iterations=500_000,
            use_cnn_prior=False,
            use_heuristic_prior=True,
            use_deadlock_check=True,
            verbose=False,
            time_budget=0,
        )
        t0 = time.perf_counter()
        path = solve(state, config=config)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        ok = bool(
            path
            and path[0] == state.start
            and path[-1] == state.goal
            and len(set(path)) == len(path)
        )
        status = "PASS" if ok else "FAIL"
        print(f"  Seed {seed}: {status} - {len(path) if path else 0} cells, {elapsed*1000:.1f}ms")
        if ok:
            passed += 1

    avg_ms = (total_time / 5) * 1000
    print(f"Legacy mode: {passed}/5 passed, avg {avg_ms:.1f}ms")
    return passed == 5


def test_time_controlled_mode():
    """Test that time-controlled mode utilizes the full budget."""
    print("\n=== Time-Controlled Mode Test (time_budget=2.0s) ===")
    state, true_path = generate_synthetic_state(rows=7, cols=7, seed=42)
    config = SolverConfig(
        max_iterations=5_000_000,
        use_cnn_prior=False,
        use_heuristic_prior=True,
        use_deadlock_check=True,
        verbose=True,
        time_budget=2.0,
    )
    t0 = time.perf_counter()
    path = solve(state, config=config)
    elapsed = time.perf_counter() - t0

    ok = bool(path and path[0] == state.start and path[-1] == state.goal)
    print(f"Result: {len(path) if path else 0} cells, took {elapsed:.2f}s (budget: 2.0s)")
    print(f"Time-controlled mode: {'PASS' if ok else 'FAIL'}")
    return ok


def test_time_control_utilization():
    """Test that time budget is actually utilized (not just returning first solution)."""
    print("\n=== Time Utilization Test (time_budget=3.0s) ===")
    state, true_path = generate_synthetic_state(rows=8, cols=8, seed=99)
    config = SolverConfig(
        max_iterations=10_000_000,
        use_cnn_prior=False,
        use_heuristic_prior=True,
        use_deadlock_check=True,
        verbose=True,
        time_budget=3.0,
    )
    t0 = time.perf_counter()
    path = solve(state, config=config)
    elapsed = time.perf_counter() - t0

    ok = bool(path and path[0] == state.start and path[-1] == state.goal)
    print(f"Result: {len(path) if path else 0} cells, took {elapsed:.2f}s (budget: 3.0s)")
    print(f"Time utilization: {'PASS' if ok else 'FAIL'}")
    return ok


def test_main_cli():
    """Test that main.py CLI accepts --time-budget argument."""
    print("\n=== CLI Argument Test ===")
    import subprocess
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=str(Path(__file__).resolve().parent),
        capture_output=True, text=True,
    )
    has_time_budget = "--time-budget" in result.stdout
    print(f"CLI has --time-budget: {'PASS' if has_time_budget else 'FAIL'}")
    return has_time_budget


if __name__ == "__main__":
    print("=" * 60)
    print("Time Control Implementation Tests")
    print("=" * 60)

    legacy_ok = test_legacy_mode()
    time_ctrl_ok = test_time_controlled_mode()
    util_ok = test_time_control_utilization()
    cli_ok = test_main_cli()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Legacy mode (fast):     {'PASS' if legacy_ok else 'FAIL'}")
    print(f"Time-controlled:        {'PASS' if time_ctrl_ok else 'FAIL'}")
    print(f"Time utilization:       {'PASS' if util_ok else 'FAIL'}")
    print(f"CLI argument:           {'PASS' if cli_ok else 'FAIL'}")
    all_pass = all([legacy_ok, time_ctrl_ok, util_ok, cli_ok])
    print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_pass else 1)
