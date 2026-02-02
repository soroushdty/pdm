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

    This function prefers the 'DIR_INPUT' key for the input dataset. For backward
    compatibility it still accepts 'dataset' and will copy 'DIR_INPUT' into
    'dataset' when present so the rest of the code can continue to use
    cfg['dataset'].
    """
    # required-ish keys: prefer DIR_INPUT but accept dataset for compatibility
    if "DIR_INPUT" in cfg:
        # mirror into 'dataset' so downstream code that expects 'dataset' keeps working
        cfg.setdefault("dataset", cfg["DIR_INPUT"]) 
    elif "dataset" not in cfg:
        raise ValueError("Config is missing required key: 'DIR_INPUT' (path to xlsx) or 'dataset'.")

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