"""Imbalance-handling strategies.

Two options, so the pipeline can A/B them:

1. SMOTE — synthesizes new minority-class points by interpolating between
   real minority neighbors. Applied ONLY to X_train/y_train, never to
   validation or test data (evaluating on synthetic points would be
   meaningless — you'd be scoring the model on data it partly invented).

2. scale_pos_weight — no resampling at all; instead XGBoost's loss function
   is reweighted so minority-class errors count more. Cheaper, doesn't
   inflate the dataset, and sometimes matches SMOTE's performance — worth
   comparing rather than assuming SMOTE is always better.
"""

import pandas as pd


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float,
    k_neighbors: int,
    random_state: int,
):
    from imblearn.over_sampling import SMOTE

    pos_rate_before = y_train.mean()
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(
        f"[resampling] SMOTE: {len(X_train):,} rows (pos={pos_rate_before:.4%}) "
        f"-> {len(X_res):,} rows (pos={y_res.mean():.4%})"
    )
    return X_res, y_res


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Ratio of negative to positive examples, XGBoost's recommended
    starting point for scale_pos_weight under imbalance."""
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    weight = n_neg / max(n_pos, 1)
    print(f"[resampling] scale_pos_weight = {weight:.1f} (no oversampling)")
    return weight
