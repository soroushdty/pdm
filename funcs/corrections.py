"""Apply JSON item renaming and final dedup."""

from __future__ import annotations

import logging
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

def apply_json_corrections(df: pd.DataFrame, json_map: Dict[str, Dict[str, str]], item_col: str) -> pd.DataFrame:
    if item_col not in df.columns:
        raise ValueError(f"Column '{item_col}' missing from DataFrame.")

    total_groups = len(json_map)
    total_rows_updated = 0

    for new_item_name, index_map in json_map.items():
        if not index_map:
            continue
        try:
            target_indices = [int(k) for k in index_map.keys()]
        except ValueError as e:
            logger.error("Failed to convert indices to integers for item '%s': %s", new_item_name, e)
            continue

        valid_indices = [idx for idx in target_indices if idx in df.index]
        if len(valid_indices) != len(target_indices):
            missing = set(target_indices) - set(valid_indices)
            logger.warning("Indices %s for item '%s' not found in DataFrame; skipped.", missing, new_item_name)

        if valid_indices:
            df.loc[valid_indices, item_col] = new_item_name
            total_rows_updated += len(valid_indices)

    logger.info("Correction complete. Groups processed: %d. Rows updated: %d", total_groups, total_rows_updated)
    return df

def final_deduplication(df: pd.DataFrame, patient_col: str, item_col: str, classes_list: list[str]) -> pd.DataFrame:
    agg_dict: Dict[str, Any] = {c: 'max' for c in classes_list if c in df.columns}
    other_cols = [c for c in df.columns if c not in classes_list and c not in [patient_col, item_col]]
    for c in other_cols:
        agg_dict[c] = 'first'
    return df.groupby([patient_col, item_col], as_index=False).agg(agg_dict)
