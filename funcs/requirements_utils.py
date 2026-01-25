"""Utilities for managing Python package requirements in Google Colab.

- Reads requirements.txt
- Installs missing packages via pip
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

def _iter_requirements_lines(requirements_path: str | Path) -> List[str]:
    path = Path(requirements_path)
    if not path.exists():
        raise FileNotFoundError(f"requirements file not found: {path}")
    reqs: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # keep simple specifiers; pip can handle them
        reqs.append(line)
    return reqs

def is_package_importable(requirement: str) -> bool:
    """Best-effort check whether a requirement is importable.

    For things like 'pandas==2.2.0', we try import name 'pandas'.
    """
    name = requirement
    for sep in ["==", ">=", "<=", "~=", ">", "<"]:
        if sep in name:
            name = name.split(sep, 1)[0].strip()
    # common pip-name vs import-name mismatches can be added here if needed
    pip_to_import = {
        "scikit-learn": "sklearn",
        "opencv-python": "cv2",
        "python-dateutil": "dateutil",
    }
    import_name = pip_to_import.get(name, name)
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ModuleNotFoundError, ValueError):
        return False

def install_missing(requirements_path: str | Path, quiet: bool = False) -> List[str]:
    """Install missing packages listed in requirements.txt. Returns installed requirements."""
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
