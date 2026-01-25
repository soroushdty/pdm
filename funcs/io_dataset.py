"""Dataset I/O."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

def read_train_test_xlsx(dataset_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    try:
        df_train = pd.read_excel(path, sheet_name='train', dtype=str, keep_default_na=False)
        df_test = pd.read_excel(path, sheet_name='test', dtype=str, keep_default_na=False)
    except ValueError as e:
        raise ValueError("Ensure the workbook contains 'train' and 'test' sheets.") from e

    # Add tracking columns (1-based)
    df_train["_original_row_1based"] = list(range(1, len(df_train) + 1))
    df_test["_original_row_1based"] = list(range(1, len(df_test) + 1))

    return df_train, df_test
