"""Utilities for managing Python package requirements in Google Colab.

- Reads requirements.txt
- Checks if module already present
- If not, try to import module
- If not, try to install missing packages via pip
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

def _iter_requirements_lines(requirements_path: str | Path) -> List[str]:
    """
    Parse requirements.txt file
    """
    path = Path(requirements_path)
    if not path.exists():
        raise FileNotFoundError(f"requirements file not found: {path}")
    reqs: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return reqs


def is_package_installed(requirements_path: str | Path) -> bool:
    """
    Check whether all packages listed in requirements.txt are installed.
    """
    reqs = _iter_requirements_lines(requirements_path)
    for r in reqs:
        if not is_package_importable(r):
            return False
    return True


def is_package_importable(requirement: str) -> bool:
    """
    Check whether a requirement is importable.
    """
    for sep in ["==", ">=", "<=", "~=", ">", "<"]:
        if sep in requirement:
            name = requirement.split(sep, 1)[0].strip()
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def install_missing(requirements_path: str | Path, quiet: bool = False) -> List[str]:
    """
    Install missing packages listed in requirements.txt. Returns installed requirements."""
    reqs = _iter_requirements_lines(requirements_path)
    missing = [r for r in reqs if not is_package_importable(r)]
    if not missing:
        return []

    pip_cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        pip_cmd.append("-q")
    pip_cmd.extend(missing)

    subprocess.check_call(pip_cmd)
    return missing
