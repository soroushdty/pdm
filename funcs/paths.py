"""Filesystem helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

def phoenix_time() -> datetime:
    """Return current time in America/Phoenix (no DST) using fixed UTC-7 offset."""
    utc_now = datetime.now(timezone.utc)
    return utc_now - timedelta(hours=7)

def make_run_dir(base_output_path: str | Path) -> Path:
    base = Path(base_output_path)
    base.mkdir(parents=True, exist_ok=True)
    ts = phoenix_time().strftime('%H-%M_%b-%d').upper()
    run_dir = base / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
