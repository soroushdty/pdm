"""Dataset I/O."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_train_test_csv_dir(dataset_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read train.csv and test.csv from a directory.

    Args:
        dataset_dir: Path to a directory containing 'train.csv' and 'test.csv'.

    Returns:
        Tuple of (df_train, df_test) where both are pandas.DataFrame with
        an added column '_original_row_1based' (1-based row index).

    Raises:
        FileNotFoundError: if directory or expected CSV files are missing.
        ValueError: if CSV reading fails for other reasons.
    """
    path = Path(dataset_dir)
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found at: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Expected a directory for dataset_path but got a file: {path}")

    train_csv = path / "train.csv"
    test_csv = path / "test.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing expected file: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing expected file: {test_csv}")

    try:
        # Keep behaviour similar to previous function: read everything as str and don't convert NA
        df_train = pd.read_csv(train_csv, dtype=str, keep_default_na=False)
        df_test = pd.read_csv(test_csv, dtype=str, keep_default_na=False)
    except Exception as e:
        raise ValueError("Failed to read train.csv and test.csv from the provided directory.") from e

    # Add tracking columns (1-based)
    df_train["_original_row_1based"] = list(range(1, len(df_train) + 1))
    df_test["_original_row_1based"] = list(range(1, len(df_test) + 1))

    return df_train, df_test