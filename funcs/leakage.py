"""Remove potential data leakage between train and test based on overlapping standardized item names."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

def remove_data_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_json_path: str | Path,
    test_json_path: str | Path,
    item_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_map = json.loads(Path(train_json_path).read_text(encoding="utf-8"))
    test_map = json.loads(Path(test_json_path).read_text(encoding="utf-8"))

    train_keys = set(train_map.keys())
    test_keys = set(test_map.keys())
    overlapping = train_keys.intersection(test_keys)

    logger.info("Unique Items in Train JSON: %d", len(train_keys))
    logger.info("Unique Items in Test JSON: %d", len(test_keys))
    logger.info("Overlapping Items found: %d", len(overlapping))

    rows_before = len(test_df)
    mask = test_df[item_col].isin(overlapping)
    test_df_clean = test_df[~mask].copy()
    logger.info("Test rows before: %d | removed: %d | after: %d",
                rows_before, rows_before - len(test_df_clean), len(test_df_clean))
    return train_df, test_df_clean
