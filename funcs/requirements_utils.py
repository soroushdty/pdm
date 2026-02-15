"""Utilities for managing Python package requirements in Google Colab.

- Reads requirements.txt
- Checks if module already present
- If not, try to import module
# - If not, try to install missing packages via pip
"""

from __future__ import annotations
import sys
import logging
import subprocess
import importlib
from pathlib import Path
from typing import List


def _iter_requirements(requirements_path: str | Path) -> List[str]:
    """
    Read requirements.txt and return requirements.
    """
    if not requirements_path.exists():
        raise FileNotFoundError(f"requirements file not found at {requirements_path}")
    reqs = []
    for raw in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            reqs.append(line)
            logging.info(f"Package '{line}' is required.")
        
    return reqs
   
def requirements_utils (
    requirements_path: str | Path, quiet: bool = True,) -> None:
    """
    Parses and installs packages from requirements.txt
    Args:
        requirements_path (str | Path): Path to requirements.txt file.
        quiet (bool): If True, suppresses pip output.
    Returns:
        List of installed packages.
    """
    reqs = _iter_requirements(requirements_path)
    pip_cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        pip_cmd.append("-q")
    installed = []
    for pkg in reqs:
        logging.info(f"Installing required package: {pkg}")
        installed.append(pkg)
        pip_cmd.extend(reqs)
        subprocess.check_call(pip_cmd)
    return installed