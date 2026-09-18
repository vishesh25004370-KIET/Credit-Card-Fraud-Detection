"""Feature engineering + splitting.

Design choice: splits happen BEFORE any resampling or scaling, and any
transform that "learns" something from data (imputation values, category
encodings) is fit only on the training fold. This is what keeps SMOTE (and
everything else) leak-free — see resampling.py and the README's
"Pitfalls avoided" section.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    feature_names: list


def engineer_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Dataset-agnostic feature engineering.

    Handles the common `creditcard.csv` schema (Time in seconds, Amount)
    and generic categorical columns, without assuming either is present.
    """
    df = df.copy()

    # Time -> cyclical hour-of-day features (creditcard.csv gives seconds
    # since the first transaction in the dataset, spanning ~2 days)
    if "Time" in df.columns:
        seconds_in_day = 24 * 60 * 60
        hour_of_day = (df["Time"] % seconds_in_day) / 3600.0
        df["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24)
        df = df.drop(columns=["Time"])

    # Amount -> log1p to tame the heavy right skew typical of transaction
    # amounts; keep the raw amount too since tree models can use both
    if "Amount" in df.columns:
        df["Amount_log1p"] = np.log1p(df["Amount"])

    # generic categorical encoding for anything object-typed (IEEE-CIS has
    # several, e.g. ProductCD, card4, card6, P_emaildomain)
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns
                if c != target_col]
    for c in cat_cols:
        df[c] = df[c].astype("category").cat.codes.replace(-1, np.nan)

    # numeric imputation: median is robust to the outliers common in
    # financial features; fit-on-train happens naturally since this runs
    # per-split in split_data() below, not on the full df up front
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> SplitData:
    df = engineer_features(df, target_col)

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    feature_names = list(X.columns)

    # stratified split preserves the (tiny) fraud rate in every split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        stratify=y_train_full,
        random_state=random_state,
    )

    # median-impute using ONLY training-fold statistics, applied to all
    # three splits — this is the leak-safe way to handle missing values
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)

    print(
        f"[preprocessing] train={len(X_train):,} "
        f"(pos={y_train.mean():.4%})  val={len(X_val):,} "
        f"(pos={y_val.mean():.4%})  test={len(X_test):,} "
        f"(pos={y_test.mean():.4%})"
    )

    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
