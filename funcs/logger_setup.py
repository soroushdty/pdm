"""Logging setup."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

def _phoenix_time_str() -> str:
    utc_now = datetime.now(timezone.utc)
    phx_time = utc_now - timedelta(hours=7)
    return phx_time.strftime("%H:%M:%S %m-%d-%y")

class PhxFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):  # noqa: D401
        return _phoenix_time_str()

def setup_logger(log_path: str | Path) -> logging.Logger:
    """Create a root logger with console INFO and file DEBUG."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear handlers to avoid duplicates on re-run in notebooks
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = PhxFormatter('[%(asctime)s] %(levelname)s: %(message)s')

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    file_handler = logging.FileHandler(str(log_path))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    root_logger.info("Logging initialized at: %s", log_path)
    return root_logger
