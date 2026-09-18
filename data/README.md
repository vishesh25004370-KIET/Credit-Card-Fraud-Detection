# Data

This directory is intentionally empty in git (see `.gitignore`) — Kaggle
does not permit redistributing the dataset, and IEEE-CIS is too large for
a repo anyway.

> **Included locally:** `data/creditcard.csv` in this download is a
> **5,000-row sample** of the full 284,807-row Kaggle `creditcardfraud`
> file (60 fraud / 4,940 legitimate, same columns: `Time`, `V1..V28`,
> `Amount`, `Class`). It's `.gitignore`d, so it won't get pushed to GitHub
> — that's intentional. It's fine for smoke-testing the pipeline, but too
> small and too easy to report real results from (the test split only has
> ~12 fraud rows). Download the full file below before reporting numbers.

## Option A — `creditcardfraud` (simplest, recommended to start)

https://www.kaggle.com/mlg-ulb/creditcardfraud

284,807 transactions, 492 fraudulent (0.17%). Already PCA-anonymized to
`V1..V28`, plus `Time`, `Amount`, `Class` (1 = fraud).

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```

You should end up with `data/creditcard.csv`. Run with:

```bash
python main.py --data data/creditcard.csv --target Class
```

## Option B — IEEE-CIS Fraud Detection (larger, more realistic)

https://www.kaggle.com/c/ieee-fraud-detection

Two files, `train_transaction.csv` and `train_identity.csv`, joined on
`TransactionID`. Label column is `isFraud`. Requires a small join step
before `main.py` — see the notebook (`notebooks/01_fraud_detection_pipeline.ipynb`,
first two cells) for the merge code, then save the merged file as
`data/ieee_merged.csv` and run:

```bash
python main.py --data data/ieee_merged.csv --target isFraud
```

## Option C — no Kaggle account, just want to run the code

```bash
python src/make_synthetic_data.py --out data/synthetic_creditcard.csv
python main.py --data data/synthetic_creditcard.csv --target Class
```

Synthetic data only — fine for checking the pipeline runs, not for
reporting results.
