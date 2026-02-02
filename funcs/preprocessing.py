"""End-to-end preprocessing pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import pandas as pd

from .io_dataset import read_train_test_csv_dir
from .merge_physician_rows import merge_rows_by_patient_item
from .item_standardization import make_items_auto_std
from .corrections import apply_json_corrections, final_deduplication
from .leakage import remove_data_leakage

logger = logging.getLogger(__name__)

def run(cfg: dict) -> dict:
    input_dir = Path.cwd() / cfg["DIR_INPUT"]
    output_path = Path.cwd() / cfg["DIR_OUTPUT"]
    patient_col = cfg["patient_col"]
    physician_col = cfg["physician_col"]
    item_col = cfg["item_col"]
    classes_list = cfg.get("classes", [])

    # Read train/test from CSV using the helper
    df_train, df_test = read_train_test_csv_dir(input_dir)

    merged_train, count_train = merge_rows_by_patient_item(
        df_train, patient_col, physician_col, item_col, classes_list, split_name="train"
    )
    merged_test, count_test = merge_rows_by_patient_item(
        df_test, patient_col, physician_col, item_col, classes_list, split_name="test"
    )

    # reset index so JSON uses stable 0..n-1 indices
    merged_train = merged_train.reset_index(drop=True)
    merged_test = merged_test.reset_index(drop=True)

    train_map_path = output_path / "merge_map_train.json"
    test_map_path = output_path / "merge_map_test.json"

    make_items_auto_std(
        merged_train, train_map_path, item_col=item_col, patient_col=patient_col,
        similarity_threshold=float(cfg.get("similarity_threshold", 0.82)),
        index_base=0,
    )
    make_items_auto_std(
        merged_test, test_map_path, item_col=item_col, patient_col=patient_col,
        similarity_threshold=float(cfg.get("similarity_threshold", 0.82)),
        index_base=0,
    )

    corrections_map_train = json.loads(train_map_path.read_text(encoding="utf-8"))
    corrections_map_test = json.loads(test_map_path.read_text(encoding="utf-8"))

    standard_train = apply_json_corrections(merged_train, corrections_map_train, item_col=item_col)
    standard_test = apply_json_corrections(merged_test, corrections_map_test, item_col=item_col)

    standard_train = final_deduplication(standard_train, patient_col, item_col, classes_list)
    standard_test = final_deduplication(standard_test, patient_col, item_col, classes_list)

    final_train, final_test = remove_data_leakage(
        standard_train, standard_test, train_map_path, test_map_path, item_col=item_col
    )

    train_csv = output_path / "train.csv"
    test_csv = output_path / "test.csv"
    final_train.to_csv(train_csv, index=False)
    final_test.to_csv(test_csv, index=False)

    return {
        "merged_counts": {"train": count_train, "test": count_test},
        "paths": {
            "output_dir": str(output_path),
            "train_map": str(train_map_path),
            "test_map": str(test_map_path),
            "train_csv": str(train_csv),
            "test_csv": str(test_csv),
        },
        "shapes": {
            "train_raw": df_train.shape,
            "test_raw": df_test.shape,
            "train_final": final_train.shape,
            "test_final": final_test.shape,
        }
    }
