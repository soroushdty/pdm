"""Utilities for managing Python package requirements in Google Colab.

- Reads requirements.txt
- Checks if module already present
- If not, try to import module
- If not, try to install missing packages via pip
"""

from __future__ import annotations
import logging
import subprocess
import sys
from pathlib import Path
from typing import List
import importlib

def _iter_requirements(requirements_path: str | Path) -> List[str]:
    """
    Check whether any of packages listed in requirements.txt already exists in work environment
    """
    if not requirements_path.exists():
        raise FileNotFoundError(f"requirements file not found: {requirements_path}")
    reqs: List[str] = []
    for raw in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            sanitized = line.split("==")[0].split(">=")[0].split("<=")[0].strip().replace("-", "_")
            if sanitized not in sys.modules:
                reqs.append(line)
        
    return reqs


def is_package_importable(package_name: str) -> bool:
    try:
        importlib.import_module(package_name)
        logging.info(f"Package '{package_name}' is already imported.")
        return True
    except ModuleNotFoundError:
        return False

def requirements_utils(requirements_path: str | Path, quiet: bool = False) -> List[str]:
    """
    Imports / installs missing packages listed in requirements.txt
    """
    reqs = _iter_requirements(requirements_path)
    missing = [ ]
    for r in reqs:
        logging.info(f"Package '{r}' is missing and will be installed.")
        if is_package_importable(r):
            continue
        else:
            missing.append(r)
   
    pip_cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        pip_cmd.append("-q")
    pip_cmd.extend(missing)
    subprocess.check_call(pip_cmd)
    logging.info("The following missing packages have been installed: " + ", ".join(missing))
