# About these output files

The files in this folder (`metrics.json`, `confusion_matrix.png`,
`pr_curve.png`, `roc_curve.png`, `feature_importance.png`) were generated
from `data/creditcard.csv` — a **5,000-row sample**, not the full
284,807-row dataset — using a `scikit-learn` stand-in
(`HistGradientBoostingClassifier` + a hand-written k-NN oversampler), not
the real `main.py` pipeline (real `XGBoost` + `imblearn.SMOTE`).

Two things not to do with them:
- Don't cite `pr_auc: 1.0` / `roc_auc: 1.0` as a real result — with only
  ~12 fraud cases in the test split, perfect separation reflects the
  sample being too small and too easy, not a good model.
- Don't present these as XGBoost output — they came from a different
  model, generated in an environment where XGBoost couldn't be installed.

## Regenerate the real thing

```bash
pip install -r requirements.txt
python main.py --data data/creditcard.csv --target Class      # your sample, or
python main.py --data data/creditcard_full.csv --target Class # the full Kaggle file
```

That overwrites everything in this folder (including this note) with
real `main.py` output: `metrics.json`, the four plots above, plus
`threshold_sweep.png` and the saved model `xgb_fraud_model.json`. Delete
this file once you've done that.
