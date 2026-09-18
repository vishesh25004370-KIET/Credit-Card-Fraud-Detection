"""Central place for pipeline hyperparameters so every module reads from
one source instead of scattering magic numbers across files."""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class PipelineConfig:
    # data
    data_path: str
    target_col: str = "Class"
    test_size: float = 0.2
    val_size: float = 0.2  # fraction of the *training* split held out for
    # early stopping / threshold tuning (never touches test data)
    random_state: int = 42

    # resampling
    resample_strategy: Literal["smote", "scale_pos_weight", "none"] = "smote"
    smote_k_neighbors: int = 5
    # target minority:majority ratio after SMOTE. 1.0 = fully balanced;
    # lower values (e.g. 0.1) are often better than full balancing because
    # they avoid manufacturing too many synthetic points in sparse regions.
    smote_sampling_strategy: float = 0.1

    # model
    n_estimators: int = 1000
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 50
    eval_metric: str = "aucpr"

    # threshold tuning
    threshold_metric: Literal["f1", "recall_at_precision", "cost"] = "f1"
    min_precision: float = 0.5  # used when threshold_metric == recall_at_precision
    fp_cost: float = 1.0        # used when threshold_metric == cost
    fn_cost: float = 10.0       # used when threshold_metric == cost

    # output
    output_dir: str = "outputs"
    use_shap: bool = False

    def __post_init__(self):
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be in (0, 1)")
        if not 0 < self.val_size < 1:
            raise ValueError("val_size must be in (0, 1)")
        if self.resample_strategy not in ("smote", "scale_pos_weight", "none"):
            raise ValueError(f"unknown resample_strategy: {self.resample_strategy}")
