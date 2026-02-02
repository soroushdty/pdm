"""Config loading"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config JSON must be an object (dict) at the top level.")

    return cfg
