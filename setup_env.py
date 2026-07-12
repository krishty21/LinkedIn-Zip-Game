"""
setup_env.py
============
Phase 1 - Automated Environment Provisioning.

This script is fully self-contained and idempotent. Running it will:

  1. Create a local virtual environment at ``./venv`` (skipped if it already
     exists and is healthy).
  2. (Re)generate ``requirements.txt`` with the exact pin set required by the
     project: playwright, torch, torchvision, opencv-python, numpy, pillow.
  3. Install every requirement into the freshly created venv using ``pip``.
  4. Execute ``playwright install chromium`` so the headless browser binaries
     are downloaded into the venv's cache directory.

The script is safe to re-run: each phase checks whether its target state is
already satisfied and short-circuits when possible.

Usage
-----
    python setup_env.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent
VENV_DIR: Path = PROJECT_ROOT / "venv"
REQUIREMENTS_FILE: Path = PROJECT_ROOT / "requirements.txt"

# The exact pin set requested in the specification. Versions are kept as
# lower bounds so that future-compatible patches are picked up automatically.
REQUIRED_PACKAGES: List[str] = [
    "playwright>=1.40.0",
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "opencv-python>=4.8.0",
    "numpy>=1.24.0",
    "pillow>=10.0.0",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Prefix every log line so the user can tell which phase produced it."""
    print(f"[setup_env] {msg}", flush=True)


def _venv_python() -> Path:
    """Return the absolute path to the venv's python interpreter."""
    if os.name == "nt":  # Windows
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _venv_pip() -> List[str]:
    """Return the command prefix used to invoke pip inside the venv."""
    return [str(_venv_python()), "-m", "pip"]


def _run(cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
    """Thin wrapper around subprocess.run with sensible defaults."""
    _log("$ " + " ".join(cmd))
    kwargs.setdefault("check", True)
    kwargs.setdefault("text", True)
    return subprocess.run(cmd, **kwargs)


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

def create_virtualenv() -> None:
    """Create the venv if it does not already exist."""
    if _venv_python().exists():
        _log(f"Virtual environment already exists at {VENV_DIR} - skipping.")
        return

    _log(f"Creating virtual environment at {VENV_DIR} ...")
    builder = venv.EnvBuilder(
        with_pip=True,
        upgrade_deps=True,
        symlinks=True,
        clear=True,
    )
    builder.create(str(VENV_DIR))
    _log("Virtual environment created successfully.")


def write_requirements() -> None:
    """(Re)write requirements.txt from the canonical pin set."""
    _log(f"Writing {REQUIREMENTS_FILE} ...")
    REQUIREMENTS_FILE.write_text(
        "\n".join(REQUIRED_PACKAGES) + "\n",
        encoding="utf-8",
    )


def install_requirements() -> None:
    """Install every requirement into the venv."""
    _log("Upgrading pip / setuptools / wheel ...")
    _run(_venv_pip() + ["install", "--upgrade", "pip", "setuptools", "wheel"])

    _log("Installing project requirements ...")
    _run(
        _venv_pip() + ["install", "-r", str(REQUIREMENTS_FILE)],
    )
    _log("All Python dependencies installed.")


def install_playwright_browsers() -> None:
    """Download the Chromium binary required by Playwright."""
    _log("Running `playwright install chromium` ...")
    _run([str(_venv_python()), "-m", "playwright", "install", "chromium"])
    _log("Chromium browser binary installed.")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    """Run every provisioning phase sequentially."""
    try:
        create_virtualenv()
        write_requirements()
        install_requirements()
        install_playwright_browsers()
    except subprocess.CalledProcessError as exc:
        _log(f"ERROR: command failed with code {exc.returncode}: {exc}")
        return exc.returncode
    except OSError as exc:
        _log(f"ERROR: OS-level failure: {exc}")
        return 1

    _log("Environment provisioning complete.")
    _log(f"  - venv:        {VENV_DIR}")
    _log(f"  - requirements:{REQUIREMENTS_FILE}")
    _log(f"  - python:      {_venv_python()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
