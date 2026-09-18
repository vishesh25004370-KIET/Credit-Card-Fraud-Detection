#!/usr/bin/env python
"""End-to-end credit card fraud detection pipeline.

    python main.py --data data/creditcard.csv --target Class

See README.md for the full flag list and methodology notes.
"""

import argparse
from pathlib import Path

from src.config import PipelineConfig
from src.data_loader import load_transactions
from src.preprocessing import split_data
from src.resampling import apply_smote, compute_scale_pos_weight
from src.model import train_xgboost
from src.threshold_tuning import tune_threshold
from src.evaluate import (
    evaluate_on_test,
    plot_feature_importance,
    plot_shap_summary,
    plot_threshold_sweep,
)


def parse_args() -> PipelineConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="path to transactions CSV")
    p.add_argument("--target", default="Class", help="label column name")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--resample",
        default="smote",
        choices=["smote", "scale_pos_weight", "none"],
    )
    p.add_argument("--smote-sampling-strategy", type=float, default=0.1)
    p.add_argument(
        "--threshold-metric",
        default="f1",
        choices=["f1", "recall_at_precision", "cost"],
    )
    p.add_argument("--min-precision", type=float, default=0.5)
    p.add_argument("--fp-cost", type=float, default=1.0)
    p.add_argument("--fn-cost", type=float, default=10.0)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--shap", action="store_true", help="also compute a SHAP summary plot")
    args = p.parse_args()

    return PipelineConfig(
        data_path=args.data,
        target_col=args.target,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        resample_strategy=args.resample,
        smote_sampling_strategy=args.smote_sampling_strategy,
        threshold_metric=args.threshold_metric,
        min_precision=args.min_precision,
        fp_cost=args.fp_cost,
        fn_cost=args.fn_cost,
        output_dir=args.output_dir,
        use_shap=args.shap,
    )


def run(cfg: PipelineConfig):
    df = load_transactions(cfg.data_path, cfg.target_col)
    split = split_data(
        df, cfg.target_col, cfg.test_size, cfg.val_size, cfg.random_state
    )

    scale_pos_weight = 1.0
    X_train, y_train = split.X_train, split.y_train

    if cfg.resample_strategy == "smote":
        X_train, y_train = apply_smote(
            split.X_train,
            split.y_train,
            sampling_strategy=cfg.smote_sampling_strategy,
            k_neighbors=cfg.smote_k_neighbors,
            random_state=cfg.random_state,
        )
    elif cfg.resample_strategy == "scale_pos_weight":
        scale_pos_weight = compute_scale_pos_weight(split.y_train)
    # "none" -> leave as-is, no reweighting (useful as a worst-case baseline)

    model = train_xgboost(
        X_train, y_train, split.X_val, split.y_val, cfg, scale_pos_weight
    )

    y_val_proba = model.predict_proba(split.X_val)[:, 1]
    threshold_result = tune_threshold(split.y_val, y_val_proba, cfg)
    plot_threshold_sweep(threshold_result.sweep, threshold_result.threshold, cfg.output_dir)

    metrics = evaluate_on_test(
        model, split.X_test, split.y_test, threshold_result.threshold, cfg.output_dir
    )

    plot_feature_importance(model, split.feature_names, cfg.output_dir)
    if cfg.use_shap:
        plot_shap_summary(model, split.X_test.sample(min(1000, len(split.X_test)),
                                                       random_state=cfg.random_state),
                           cfg.output_dir)

    model_path = Path(cfg.output_dir) / "xgb_fraud_model.json"
    model.save_model(model_path)
    print(f"[main] saved model to {model_path}")
    print(f"[main] all outputs written to {cfg.output_dir}/")

    return metrics


if __name__ == "__main__":
    run(parse_args())
