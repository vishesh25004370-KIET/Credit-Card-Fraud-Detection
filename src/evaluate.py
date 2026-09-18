"""Final test-set evaluation, plots, and feature importance / interpretation.

The test set is touched exactly once, here, after the threshold has already
been chosen on the validation set -- this keeps the reported numbers honest.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_on_test(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    output_dir: str,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "precision_at_threshold": precision_score(y_test, y_pred, zero_division=0),
        "recall_at_threshold": recall_score(y_test, y_pred, zero_division=0),
        "f1_at_threshold": f1_score(y_test, y_pred, zero_division=0),
        "threshold": threshold,
        "n_test": int(len(y_test)),
        "n_fraud_test": int(y_test.sum()),
    }

    print("[evaluate] test-set metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    _plot_confusion_matrix(y_test, y_pred, out / "confusion_matrix.png")
    _plot_pr_curve(y_test, y_proba, metrics["pr_auc"], out / "pr_curve.png")
    _plot_roc_curve(y_test, y_proba, metrics["roc_auc"], out / "roc_curve.png")

    return metrics


def plot_threshold_sweep(sweep: pd.DataFrame, chosen_threshold: float, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sweep["threshold"], sweep["precision"], label="Precision")
    ax.plot(sweep["threshold"], sweep["recall"], label="Recall")
    ax.plot(sweep["threshold"], sweep["f1"], label="F1", linestyle="--")
    ax.axvline(chosen_threshold, color="black", linestyle=":", label=f"chosen={chosen_threshold:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold sweep (validation set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "threshold_sweep.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_names: list, output_dir: str, top_n: int = 20):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    booster = model.get_booster()
    importance_types = ["gain", "weight"]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, top_n * 0.3)))

    for ax, imp_type in zip(axes, importance_types):
        scores = booster.get_score(importance_type=imp_type)
        # XGBoost names features fN by index if X isn't a DataFrame at fit
        # time; map back to real names defensively
        named = {}
        for k, v in scores.items():
            if k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                name = feature_names[idx] if idx < len(feature_names) else k
            else:
                name = k
            named[name] = v

        series = pd.Series(named).sort_values(ascending=True).tail(top_n)
        ax.barh(series.index, series.values)
        ax.set_title(f"Feature importance ({imp_type})")
        ax.set_xlabel(imp_type)

    fig.suptitle(
        "Gain vs. weight importance can disagree — gain reflects average "
        "improvement per split, weight is just split count, which is "
        "biased toward high-cardinality features.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=150)
    plt.close(fig)


def plot_shap_summary(model, X_sample: pd.DataFrame, output_dir: str):
    """Optional, more reliable alternative to native XGBoost importance.
    Only called if --shap is passed and the `shap` package is installed.
    """
    try:
        import shap
    except ImportError:
        print("[evaluate] shap not installed, skipping SHAP summary plot "
              "(pip install shap)")
        return

    out = Path(output_dir)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    fig.tight_layout()
    fig.savefig(out / "shap_summary.png", dpi=150)
    plt.close(fig)


# --- plot helpers -----------------------------------------------------

def _plot_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Fraud", "Fraud"],
        yticklabels=["Not Fraud", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_pr_curve(y_true, y_proba, pr_auc, path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    baseline = np.mean(y_true)
    ax.axhline(baseline, color="gray", linestyle="--", label=f"baseline = {baseline:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_roc_curve(y_true, y_proba, roc_auc, path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
