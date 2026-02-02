"""Config loading and validation."""

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

def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate/normalize the config in-place and return it.

    'DIR_INPUT' is required. The legacy key 'dataset' is deprecated and NO
    backward-compatibility is provided here — treat configs that only provide
    'dataset' as invalid (i.e. behave as if 'dataset' never existed).
    """
    # required key: DIR_INPUT only
    if "DIR_INPUT" not in cfg:
        raise ValueError("Config is missing required key: 'DIR_INPUT' (path to xlsx).")

    # defaults
    cfg.setdefault("patient_col", "Patient")
    cfg.setdefault("physician_col", "Physician")
    cfg.setdefault("item_col", "Item")
    cfg.setdefault("output_path", "./output")

    classes = cfg.get("classes", [])
    if isinstance(classes, str):
        classes = [c.strip() for c in classes.split(",") if c.strip()]
    if not isinstance(classes, list):
        raise ValueError("Config key 'classes' must be a list (or a comma-separated string).")
    cfg["classes"] = classes

    return cfg
