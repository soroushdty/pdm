"""Utilities for managing Python package requirements in Google Colab.

- Reads requirements.txt
- Checks if module already present
- If not, try to import module
- If not, try to install missing packages via pip
"""

from __future__ import annotations
import sys
import logging
import subprocess
import importlib
from pathlib import Path
from typing import List, Tuple

def _extract_module_name(raw: str) -> str:
    """
    Normalize a requirement or import statement to a module name.
    Supports:
      - requirements: numpy==1.26.4
      - import statements: import numpy as np
      - from-import: from x.y import z
    """
    line = raw.strip()
    if line.startswith("import "):
        remainder = line[len("import "):].strip()
        first = remainder.split(",")[0].strip()
        module = first.split(" as ")[0].strip()
        return module
    if line.startswith("from "):
        remainder = line[len("from "):].strip()
        module = remainder.split(" ")[0].strip()
        return module
    sanitized = line.split("==")[0].split(">=")[0].split("<=")[0].strip()
    return sanitized.replace("-", "_")


def _iter_requirements(requirements_path: str | Path) -> List[Tuple[str, str]]:
    """
    Read requirements.txt (or import-like lines) and return (spec, module_name).
    """
    if not requirements_path.exists():
        raise FileNotFoundError(f"requirements file not found: {requirements_path}")
    reqs: List[Tuple[str, str]] = []
    for raw in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            module_name = _extract_module_name(line)
            if module_name not in sys.modules:
                reqs.append((line, module_name))
                logging.info(f"Package '{module_name}' is missing.")
        
    return reqs


def install_missing_packages(requirements_path: str | Path):
    """
    Imports / installs missing packages listed in requirements.txt
    Returns a list of missing packages.
    """
    reqs = _iter_requirements(requirements_path)
    missing = []
    for spec, module_name in reqs:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            logging.error(f"{module_name} is not installed.")
            missing.append(spec)
    return missing
   
def requirements_utils (requirements_path: str | Path, quiet: bool = True) -> None:
    """
    Installs missing packages via pip.
    """
    missing = install_missing_packages(requirements_path)
    pip_cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        pip_cmd.append("-q")
    pip_cmd.extend(missing)
    subprocess.check_call(pip_cmd)