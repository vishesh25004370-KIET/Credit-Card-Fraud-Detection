# Credit Card Fraud Detection with XGBoost + SMOTE

End-to-end pipeline for detecting fraudulent transactions in a heavily
imbalanced dataset (fraud is typically < 0.5% of transactions). Built around
the [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
dataset (also works with the smaller [Kaggle `creditcardfraud`](https://www.kaggle.com/mlg-ulb/creditcardfraud)
dataset with minor config changes).

## Why this project is interesting

Fraud detection is the canonical "imbalanced classification" problem:

- **Accuracy is a useless metric.** A model that predicts "not fraud" for
  every transaction is ~99.8% accurate and catches zero fraud.
- **Class imbalance breaks naive training.** XGBoost's default decision
  threshold (0.5) and loss function implicitly assume balanced classes, so
  the minority (fraud) class gets ignored unless you correct for it.
- **The right threshold depends on business cost**, not on abstract model
  quality — missing a $2,000 fraudulent charge and blocking a legitimate
  $40 purchase are not equally bad mistakes.

This repo addresses all three: it evaluates with PR-AUC / recall / precision
instead of accuracy, corrects imbalance with **SMOTE** oversampling (applied
correctly, inside the CV fold — see "Pitfalls avoided" below), and **tunes
the decision threshold** against precision/recall/F1/cost curves instead of
using the default 0.5.

## Pipeline

```
Raw transactions
      │
      ▼
1. Data loading & cleaning        (src/data_loader.py)
      │
      ▼
2. Feature engineering             (src/preprocessing.py)
      │  - time-based features (hour, day)
      │  - amount transforms (log1p)
      │  - categorical encoding
      │  - train/test split (stratified, chronological option)
      ▼
3. SMOTE resampling                (src/resampling.py)
      │  - fit ONLY on the training fold, never on test data
      ▼
4. XGBoost training                (src/model.py)
      │  - scale_pos_weight as a baseline alternative to SMOTE
      │  - early stopping on validation PR-AUC
      ▼
5. Threshold tuning                (src/threshold_tuning.py)
      │  - sweep thresholds, optimize for F1 / recall@precision / cost
      ▼
6. Evaluation & interpretation     (src/evaluate.py)
      │  - confusion matrix, PR curve, ROC curve
      │  - XGBoost gain-based feature importance
      │  - SHAP summary plot (optional, if `shap` is installed)
      ▼
outputs/  (metrics.json, plots, saved model)
```

Run the whole thing with:

```bash
python main.py --data data/transactions.csv --target isFraud
```

Or open `notebooks/01_fraud_detection_pipeline.ipynb` for the exploratory,
cell-by-cell version with inline plots and commentary.

## Getting the data

This repo does **not** ship the dataset (Kaggle's terms prohibit
redistribution, and IEEE-CIS is several hundred MB).

1. Download from Kaggle:
   - IEEE-CIS: `kaggle competitions download -c ieee-fraud-detection`
   - or the smaller PCA-anonymized set: `kaggle datasets download -d mlg-ulb/creditcardfraud`
2. Unzip into `data/` so you have e.g. `data/train_transaction.csv` or
   `data/creditcard.csv`.
3. See `data/README.md` for the exact expected filenames/columns, and how
   to point `main.py` at whichever variant you're using.

**No Kaggle account handy / just want to see the pipeline run?**
`python src/make_synthetic_data.py` generates a synthetic, heavily
imbalanced (~0.4% positive) transactions dataset with the same shape as
`creditcard.csv` (`Time`, `Amount`, `V1`..`V20`, `Class`), so the full
pipeline runs end-to-end with no download. This is for demoing the code
only — report real numbers from the real dataset in your write-up.

## Installation

```bash
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Usage

```bash
# 1. (optional) generate a synthetic dataset to smoke-test the pipeline
python src/make_synthetic_data.py --out data/synthetic_creditcard.csv

# 2. run the full pipeline: preprocess -> SMOTE -> train -> tune -> evaluate
python main.py --data data/synthetic_creditcard.csv --target Class

# 3. inspect results
ls outputs/
#   metrics.json
#   confusion_matrix.png
#   pr_curve.png
#   roc_curve.png
#   threshold_sweep.png
#   feature_importance.png
#   xgb_fraud_model.json
```

Key flags (`python main.py --help` for the full list):

| Flag | Default | Meaning |
|---|---|---|
| `--data` | required | path to CSV |
| `--target` | `Class` | name of the label column |
| `--resample` | `smote` | `smote`, `scale_pos_weight`, or `none` |
| `--threshold-metric` | `f1` | objective for threshold search: `f1`, `recall_at_precision`, `cost` |
| `--min-precision` | `0.5` | used only when `--threshold-metric recall_at_precision` |
| `--fp-cost` / `--fn-cost` | `1` / `10` | used only when `--threshold-metric cost` |
| `--test-size` | `0.2` | held-out test fraction |
| `--random-state` | `42` | seed |

## Results (fill in after running on the real dataset)

| Metric | Value |
|---|---|
| ROC-AUC | — |
| PR-AUC (average precision) | — |
| Precision @ chosen threshold | — |
| Recall @ chosen threshold | — |
| F1 @ chosen threshold | — |
| Chosen threshold | — |

Replace this table with your actual numbers from `outputs/metrics.json`
after running on the full IEEE-CIS or `creditcard.csv` data — the synthetic
dataset numbers are not meaningful and shouldn't be reported as results.

## Pitfalls avoided (worth calling out in a write-up / interview)

1. **SMOTE fit on the whole dataset before splitting** — this leaks
   synthetic-neighbor information from the test set into training and
   inflates scores. Here, `train_test_split` always runs first; SMOTE is
   fit only on `X_train`/`y_train` (see `src/resampling.py`), and if you
   swap in cross-validation, SMOTE must live inside an `imblearn.pipeline.Pipeline`
   for the same reason.
2. **Accuracy as the headline metric** — reported but not optimized for;
   `evaluate.py` leads with PR-AUC, which is the right summary metric under
   heavy imbalance (ROC-AUC can look deceptively good).
3. **Default 0.5 threshold** — `threshold_tuning.py` sweeps thresholds
   against the validation set and lets you optimize for F1, for a minimum
   precision constraint, or for an explicit false-positive/false-negative
   cost ratio (blocking a real purchase vs. missing real fraud usually
   have very different costs).
4. **SMOTE oversampling only** — `--resample scale_pos_weight` is included
   as a comparison baseline, since class-weighting is sometimes competitive
   with SMOTE and much cheaper; the pipeline lets you A/B both.
5. **Feature importance without caveats** — `evaluate.py` prints both
   `gain` and `weight` importance types side by side, since they can
   disagree, and notes that XGBoost's native importance is biased toward
   high-cardinality features (SHAP values are offered as an optional,
   more reliable alternative).

## Project structure

```
.
├── README.md
├── requirements.txt
├── main.py                     # CLI entry point, orchestrates the pipeline
├── data/
│   └── README.md               # dataset download + layout instructions
├── notebooks/
│   └── 01_fraud_detection_pipeline.ipynb
├── src/
│   ├── config.py                # dataclass of pipeline hyperparameters
│   ├── data_loader.py           # CSV loading + schema validation
│   ├── preprocessing.py         # feature engineering + train/test split
│   ├── resampling.py            # SMOTE / scale_pos_weight helpers
│   ├── model.py                 # XGBoost training + early stopping
│   ├── threshold_tuning.py      # threshold sweep + selection strategies
│   ├── evaluate.py              # metrics, plots, feature importance
│   └── make_synthetic_data.py   # synthetic demo dataset generator
├── tests/
│   └── test_pipeline.py         # smoke tests on synthetic data
└── outputs/                     # generated artifacts (gitignored except .gitkeep)
```

## Tests

```bash
pytest tests/ -v
```

The tests run the full pipeline on a tiny synthetic dataset and check that
it doesn't crash and that outputs have sane shapes — they are not a
substitute for validating on the real dataset.

## License

MIT — see `LICENSE`.
