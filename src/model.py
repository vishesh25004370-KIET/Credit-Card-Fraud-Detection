"""XGBoost training with early stopping on validation PR-AUC.

PR-AUC (average precision), not accuracy or plain ROC-AUC, is used as the
early-stopping metric because it's far more sensitive to how well the model
separates the rare positive class -- see README "Pitfalls avoided".
"""

import xgboost as xgb
import pandas as pd

from src.config import PipelineConfig


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: PipelineConfig,
    scale_pos_weight: float = 1.0,
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        eval_metric=cfg.eval_metric,
        early_stopping_rounds=cfg.early_stopping_rounds,
        random_state=cfg.random_state,
        tree_method="hist",
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print(
        f"[model] trained {model.best_iteration + 1} trees "
        f"(best val {cfg.eval_metric}={model.best_score:.4f})"
    )
    return model
