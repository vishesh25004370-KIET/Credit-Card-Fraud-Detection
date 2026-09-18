"""Generates a synthetic, heavily imbalanced dataset shaped like Kaggle's
`creditcardfraud` (Time, Amount, V1..V20, Class) so the pipeline can be
smoke-tested with no Kaggle download. NOT a substitute for the real data --
see data/README.md.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def make_synthetic_transactions(
    n_samples: int = 50_000,
    n_features: int = 20,
    fraud_rate: float = 0.004,
    random_state: int = 42,
) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_redundant=4,
        n_clusters_per_class=3,
        weights=[1 - fraud_rate, fraud_rate],
        flip_y=0.001,
        class_sep=1.2,
        random_state=random_state,
    )

    rng = np.random.default_rng(random_state)
    df = pd.DataFrame(X, columns=[f"V{i+1}" for i in range(n_features)])
    df["Class"] = y

    # fraud amounts skew differently from legitimate ones, mimicking the
    # real dataset's pattern (fraud clusters at both very small "test"
    # charges and unusually large ones)
    amounts = np.where(
        y == 1,
        rng.choice([rng.uniform(0, 5), rng.uniform(200, 2000)], size=len(y)),
        rng.gamma(shape=2.0, scale=40.0, size=len(y)),
    )
    df["Amount"] = np.round(amounts, 2)
    df["Time"] = np.sort(rng.integers(0, 2 * 24 * 3600, size=len(y)))

    cols = ["Time", "Amount"] + [f"V{i+1}" for i in range(n_features)] + ["Class"]
    return df[cols]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/synthetic_creditcard.csv")
    parser.add_argument("--n-samples", type=int, default=50_000)
    parser.add_argument("--fraud-rate", type=float, default=0.004)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = make_synthetic_transactions(
        n_samples=args.n_samples,
        fraud_rate=args.fraud_rate,
        random_state=args.random_state,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} rows ({df['Class'].sum():.0f} fraud) to {args.out}")
