"""Load and lightly validate the transactions CSV.

Kept deliberately dumb: real feature engineering lives in preprocessing.py.
This module's only job is "get a clean DataFrame with the target column
in it, and fail loudly if the file looks wrong."
"""

from pathlib import Path

import pandas as pd


def load_transactions(data_path: str, target_col: str) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No file at {data_path}. See data/README.md for how to download "
            f"or generate a dataset."
        )

    df = pd.read_csv(path)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available columns: "
            f"{list(df.columns)}"
        )

    n_dupes = df.duplicated().sum()
    if n_dupes:
        print(f"[data_loader] dropping {n_dupes} exact duplicate rows")
        df = df.drop_duplicates().reset_index(drop=True)

    n_missing = df.isna().sum().sum()
    if n_missing:
        print(
            f"[data_loader] {n_missing} missing values found across "
            f"{(df.isna().sum() > 0).sum()} columns — handled in preprocessing"
        )

    pos_rate = df[target_col].mean()
    print(
        f"[data_loader] loaded {len(df):,} rows, "
        f"{df[target_col].sum():,.0f} positive ({pos_rate:.4%})"
    )

    return df
