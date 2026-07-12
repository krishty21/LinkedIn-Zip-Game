"""
package.py
==========
Phase 5 - Automated project packaging.

Walks the project workspace, copies the requested source files (and the
trained weights file if present) into a staging ``dist/`` directory, then
zips the staging directory into ``linkedin_zip_solver.zip``.

Heavy / regenerable artefacts are explicitly excluded:

    * ``venv/`` and ``.venv/``
    * ``__pycache__/`` directories
    * Playwright's ``.browser_data/`` cache
    * ``.pytest_cache/``, ``*.pyc`` files
    * The previously produced ``dist/`` directory (so we always ship a
      clean archive)

The resulting archive is placed in ``dist/linkedin_zip_solver.zip`` and is
safe to redistribute as a standalone deployment artefact.
"""

from __future__ import annotations

import fnmatch
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DIST_DIR: Path = PROJECT_ROOT / "dist"
ARCHIVE_NAME: str = "linkedin_zip_solver.zip"
ARCHIVE_PATH: Path = DIST_DIR / ARCHIVE_NAME

# Files we always bundle.
BUNDLE_FILES: List[str] = [
    "vision.py",
    "solver.py",
    "model.py",
    "executor.py",
    "setup_env.py",
    "package.py",
    "main.py",
    "requirements.txt",
    "README.md",
]

# Trained weights file (optional - bundled only if it exists).
WEIGHTS_FILE: str = "zip_cnn.pth"

# Glob patterns of files / directories to exclude from the archive.
EXCLUDE_PATTERNS: List[str] = [
    "venv",
    "venv/*",
    ".venv",
    ".venv/*",
    "env",
    "env/*",
    ".browser_data",
    ".browser_data/*",
    "__pycache__",
    "__pycache__/*",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    ".pytest_cache/*",
    ".mypy_cache",
    ".mypy_cache/*",
    "dist",
    "dist/*",
    "*.egg-info",
    "*.egg-info/*",
    ".git",
    ".git/*",
    ".DS_Store",
    "Thumbs.db",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[package] {msg}", flush=True)


def _is_excluded(path: Path) -> bool:
    """Return True if ``path`` matches any exclude pattern."""
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    parts = path.relative_to(PROJECT_ROOT).parts
    for pat in EXCLUDE_PATTERNS:
        # Match against the full relative path...
        if fnmatch.fnmatch(rel, pat):
            return True
        # ...and against each individual path component (so excluding
        # ``venv`` also excludes everything inside it).
        for part in parts:
            if fnmatch.fnmatch(part, pat):
                return True
    return False


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

def clean_dist() -> None:
    """Wipe the dist/ directory so each run produces a clean archive."""
    if DIST_DIR.exists():
        _log(f"Removing previous dist/ at {DIST_DIR}")
        shutil.rmtree(DIST_DIR, ignore_errors=True)
    DIST_DIR.mkdir(parents=True, exist_ok=True)


def stage_bundle() -> Path:
    """
    Copy every file in BUNDLE_FILES (plus optional weights) into dist/.

    Returns the staging directory path.
    """
    staging = DIST_DIR / "linkedin_zip_solver"
    staging.mkdir(parents=True, exist_ok=True)

    for name in BUNDLE_FILES:
        src = PROJECT_ROOT / name
        if not src.exists():
            _log(f"  ! missing: {name} (skipped)")
            continue
        dst = staging / name
        shutil.copy2(src, dst)
        _log(f"  + {name}")

    weights_src = PROJECT_ROOT / WEIGHTS_FILE
    if weights_src.exists():
        shutil.copy2(weights_src, staging / WEIGHTS_FILE)
        _log(f"  + {WEIGHTS_FILE}")
    else:
        _log(f"  ! {WEIGHTS_FILE} not found - "
             "archive will ship without trained weights (solver falls back "
             "to plain DFS).")

    return staging


def make_archive(staging: Path) -> Path:
    """Zip ``staging`` into ``dist/linkedin_zip_solver.zip``."""
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()

    _log(f"Creating archive: {ARCHIVE_PATH}")
    # Note: we do NOT apply _is_excluded() here because ``staging`` already
    # contains only the files we explicitly copied in stage_bundle(). The
    # exclude patterns are intended for *scanning the project root*, not
    # for scanning the staging directory (which lives under dist/ and would
    # otherwise match the ``dist/*`` exclusion pattern).
    with zipfile.ZipFile(ARCHIVE_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(staging.rglob("*")):
            if file_path.is_dir():
                continue
            # Store under a top-level ``linkedin_zip_solver/`` directory.
            arcname = Path("linkedin_zip_solver") / file_path.relative_to(staging)
            zf.write(file_path, arcname.as_posix())

    _log(f"Archive size: {ARCHIVE_PATH.stat().st_size / 1024:.1f} KB")
    return ARCHIVE_PATH


def cleanup_staging(staging: Path) -> None:
    """Remove the staging directory once the zip is built."""
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        clean_dist()
        staging = stage_bundle()
        make_archive(staging)
        cleanup_staging(staging)
    except OSError as exc:
        _log(f"ERROR: {exc}")
        return 1

    _log(f"Packaging complete: {ARCHIVE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
