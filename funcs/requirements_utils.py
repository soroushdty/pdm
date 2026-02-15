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
    for raw in requirements_path.read_text(encoding="urequirements_utilstf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            reqs.append(line)
            logging.info(f"Package '{line}' is required.")
        
    return reqs


def install_missing(missing: List[str])-> List[str]:
    """
    Installs missing packages listed in requirements.txt
    Returns a list of installed packages.
    """
    reqs = _iter_requirements(missing)
    imported = []
    installed = []
    for module_name in reqs:
        try:
            importlib.import_module(module_name)
            logging.info(f"{module_name} is already installed.")
            imported.append(module_name)
            logging.debug(f"Imported modules: {imported}")
        except:
            logging.error(f"{module_name} cannot be imported.")
            installed.append(module_name)
            logging.debug(f"Installed modules: {installed}")
    return installed
   
def requirements_utils (
    requirements_path: str | Path, quiet: bool = True,) -> None:
    """
    Parses and installs missing packages from requirements.txt
    Args:
        requirements_path (str | Path): Path to requirements.txt file.
        quiet (bool): If True, suppresses pip output.
    Returns:
        List of installed packages.
    """
    missing = install_missing(requirements_path)
    pip_cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        pip_cmd.append("-q")
    if missing:
        installed = []
        for pkg in missing:
            logging.info(f"Installing missing package: {pkg}")
            installed.append(pkg)
            pip_cmd.extend(missing)
            subprocess.check_call(pip_cmd)
    return installed