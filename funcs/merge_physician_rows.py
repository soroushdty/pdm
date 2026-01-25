"""Merge/deduplicate physician rows by (Patient, Item) and average class columns."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def merge_rows_by_patient_item(
    df: pd.DataFrame,
    patient_col: str,
    physician_col: str,
    item_col: str,
    classes_list: List[str],
    split_name: str,
) -> Tuple[pd.DataFrame, int]:
    present_classes = [c for c in classes_list if c in df.columns]
    missing_classes = [c for c in classes_list if c not in df.columns]
    if missing_classes:
        logger.warning("[%s] Configured classes not found and will be skipped: %s", split_name, missing_classes)
    logger.info("[%s] Class columns to average: %s", split_name, present_classes)

    # helper keys
    work = df.copy()
    work["_patient_key_"] = work[patient_col].astype(str).str.strip()
    work["_item_key_"] = work[item_col].astype(str).str.strip()
    work["_phys_orig_"] = work[physician_col].astype(str).str.strip()

    # numeric conversions for classes
    for c in present_classes:
        num_col = f"_num_{c}_"
        work[num_col] = pd.to_numeric(work[c].replace("", np.nan), errors="coerce")

    grouped = work.groupby(["_patient_key_", "_item_key_"], sort=False, as_index=False)
    logger.info("[%s] Unique (Patient, Item) pairs found: %d", split_name, grouped.ngroups)

    merged_rows = []
    merged_pairs = 0

    skip_cols = {
        patient_col, physician_col, item_col,
        "_original_row_1based",
        "_patient_key_", "_item_key_", "_phys_orig_",
    }

    for (patient_key, item_key), group in grouped:
        src_rows = group["_original_row_1based"].tolist()

        merged = {patient_col: patient_key, item_col: item_key}

        other_cols = [
            c for c in work.columns
            if c not in skip_cols and not c.startswith("_num_") and c not in present_classes
        ]

        for oc in other_cols:
            vals = [v for v in group[oc].tolist() if str(v).strip() != ""]
            merged[oc] = vals[0] if vals else group[oc].iloc[0]

        for c in present_classes:
            num_col = f"_num_{c}_"
            numeric_vals = pd.Series(group[num_col].tolist(), dtype=float)
            merged[c] = float(numeric_vals.mean()) if not numeric_vals.dropna().empty else np.nan

        merged["_merged_from_rows_1based"] = str(src_rows)
        merged_rows.append(merged)

        if len(group) > 1:
            merged_pairs += 1

    merged_df = pd.DataFrame(merged_rows)

    if not merged_df.empty:
        desired_order = [patient_col, item_col] + present_classes + [
            c for c in merged_df.columns if c not in ([patient_col, item_col] + present_classes)
        ]
        merged_df = merged_df[[c for c in desired_order if c in merged_df.columns]]

    logger.info("[%s] Rows before: %d; rows after: %d; pairs merged: %d",
                split_name, len(df), len(merged_df), merged_pairs)

    return merged_df, merged_pairs
