"""Pick a decision threshold on the VALIDATION set (never the test set,
or you leak test information into a model-selection choice), using one of
three strategies:

- "f1": maximize F1
- "recall_at_precision": maximize recall subject to precision >= min_precision
- "cost": minimize fp_cost * FP + fn_cost * FN, i.e. an explicit business
  cost ratio (e.g. missing real fraud is ~10x worse than a false alarm)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score

from src.config import PipelineConfig


@dataclass
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    sweep: pd.DataFrame  # threshold, precision, recall, f1, cost -- for plotting


def _sweep(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns len(thresholds) == len(precisions) - 1
    precisions, recalls = precisions[:-1], recalls[:-1]
    f1s = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls + 1e-12),
        0.0,
    )
    return pd.DataFrame(
        {"threshold": thresholds, "precision": precisions, "recall": recalls, "f1": f1s}
    )


def tune_threshold(
    y_val: pd.Series, y_val_proba: np.ndarray, cfg: PipelineConfig
) -> ThresholdResult:
    y_true = y_val.to_numpy()
    sweep = _sweep(y_true, y_val_proba)

    if cfg.threshold_metric == "f1":
        best_row = sweep.loc[sweep["f1"].idxmax()]

    elif cfg.threshold_metric == "recall_at_precision":
        feasible = sweep[sweep["precision"] >= cfg.min_precision]
        if feasible.empty:
            print(
                f"[threshold] WARNING: no threshold reaches precision >= "
                f"{cfg.min_precision}; falling back to best-F1 threshold"
            )
            best_row = sweep.loc[sweep["f1"].idxmax()]
        else:
            best_row = feasible.loc[feasible["recall"].idxmax()]

    elif cfg.threshold_metric == "cost":
        n_pos = y_true.sum()
        n_neg = len(y_true) - n_pos
        # FN = missed positives = recall deficit * n_pos ; FP derived from precision
        fn = (1 - sweep["recall"]) * n_pos
        # precision = TP / (TP + FP)  =>  FP = TP * (1/precision - 1)
        tp = sweep["recall"] * n_pos
        fp = tp * (1.0 / sweep["precision"].clip(lower=1e-6) - 1)
        sweep["cost"] = cfg.fp_cost * fp + cfg.fn_cost * fn
        best_row = sweep.loc[sweep["cost"].idxmin()]

    else:
        raise ValueError(f"unknown threshold_metric: {cfg.threshold_metric}")

    result = ThresholdResult(
        threshold=float(best_row["threshold"]),
        precision=float(best_row["precision"]),
        recall=float(best_row["recall"]),
        f1=float(best_row["f1"]),
        sweep=sweep,
    )
    print(
        f"[threshold] strategy={cfg.threshold_metric} -> "
        f"threshold={result.threshold:.3f} "
        f"(precision={result.precision:.3f}, recall={result.recall:.3f}, "
        f"f1={result.f1:.3f})"
    )
    return result
