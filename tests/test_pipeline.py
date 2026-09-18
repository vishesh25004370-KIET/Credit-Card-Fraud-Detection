"""Smoke tests: run the pipeline end-to-end on a small synthetic dataset
and check nothing crashes and outputs have sane shapes. These do NOT
validate model quality on real fraud data -- that requires the real
dataset (see data/README.md).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.config import PipelineConfig
from src.make_synthetic_data import make_synthetic_transactions
from src.preprocessing import split_data
from src.resampling import apply_smote, compute_scale_pos_weight
from src.model import train_xgboost
from src.threshold_tuning import tune_threshold
from src.evaluate import evaluate_on_test


@pytest.fixture(scope="module")
def synthetic_df():
    return make_synthetic_transactions(n_samples=4000, fraud_rate=0.02, random_state=0)


def test_synthetic_data_shape(synthetic_df):
    assert len(synthetic_df) == 4000
    assert "Class" in synthetic_df.columns
    assert synthetic_df["Class"].sum() > 0


def test_split_preserves_positive_rate(synthetic_df):
    split = split_data(synthetic_df, "Class", test_size=0.2, val_size=0.2, random_state=0)
    assert len(split.X_train) + len(split.X_val) + len(split.X_test) == len(synthetic_df)
    for y in (split.y_train, split.y_val, split.y_test):
        assert y.sum() > 0, "a split lost all positive examples"


def test_smote_balances_training_set(synthetic_df):
    split = split_data(synthetic_df, "Class", test_size=0.2, val_size=0.2, random_state=0)
    X_res, y_res = apply_smote(
        split.X_train, split.y_train, sampling_strategy=0.5, k_neighbors=3, random_state=0
    )
    assert len(X_res) > len(split.X_train)
    assert abs(y_res.mean() - 1 / 3) < 0.05  # sampling_strategy=0.5 -> ~33% positive


def test_full_pipeline_runs(tmp_path, synthetic_df):
    cfg = PipelineConfig(
        data_path="unused",  # loader bypassed in this test
        target_col="Class",
        n_estimators=20,  # tiny model, just checking plumbing
        early_stopping_rounds=5,
        output_dir=str(tmp_path),
        random_state=0,
    )
    split = split_data(synthetic_df, cfg.target_col, cfg.test_size, cfg.val_size, cfg.random_state)
    X_train, y_train = apply_smote(
        split.X_train, split.y_train, cfg.smote_sampling_strategy, cfg.smote_k_neighbors, cfg.random_state
    )
    model = train_xgboost(X_train, y_train, split.X_val, split.y_val, cfg)
    y_val_proba = model.predict_proba(split.X_val)[:, 1]
    threshold_result = tune_threshold(split.y_val, y_val_proba, cfg)
    metrics = evaluate_on_test(model, split.X_test, split.y_test, threshold_result.threshold, cfg.output_dir)

    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "confusion_matrix.png").exists()
