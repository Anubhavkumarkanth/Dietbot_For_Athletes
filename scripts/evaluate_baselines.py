"""Compare the shipped Random Forest with simpler baselines on the held-out test split.

The README table compares the forest with the training mean and with the
Mifflin-St Jeor formula. Neither is fitted to this data, so beating them does
not show that a forest is needed. This adds a linear regression fitted on the
same features, and puts a bootstrap interval around each difference in MAE, so
the claim can be stated at its real size.

It evaluates the committed model, not a retrained one, on the same 60/20/20
split train_models.py uses. Nothing is written to disk.

    python scripts/evaluate_baselines.py
"""

import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from train_models import (
    CALORIE_MODEL,
    FEATURES,
    INTAKE_CSV,
    METRICS_FILE,
    TARGET,
    _split,
    mifflin_st_jeor,
)

BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 2026


def bootstrap_mae_difference(y_true, better, worse, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """95% interval for MAE(better) - MAE(worse), resampling the test rows.

    The same resampled rows are used for both models, so the interval reflects
    how consistently one beats the other on the same people. A negative number
    means `better` has the lower error.
    """
    rng = np.random.default_rng(seed)
    errors_better = np.abs(y_true - better)
    errors_worse = np.abs(y_true - worse)
    rows = rng.integers(0, len(y_true), size=(resamples, len(y_true)))
    differences = errors_better[rows].mean(axis=1) - errors_worse[rows].mean(axis=1)
    low, high = np.percentile(differences, [2.5, 97.5])
    return float(low), float(high), float((differences < 0).mean())


def main():
    df = pd.read_csv(INTAKE_CSV)
    X, y = df[FEATURES], df[TARGET]
    X_train, _, X_test, y_train, _, y_test = _split(X, y)
    y_true = y_test.to_numpy()

    forest = joblib.load(CALORIE_MODEL)
    linear = LinearRegression().fit(X_train, y_train)

    predictions = {
        "mean of training set": np.full(len(y_test), y_train.mean()),
        "Mifflin-St Jeor x 1.375": mifflin_st_jeor(X_test),
        "linear regression (fitted)": linear.predict(X_test),
        "Random Forest (shipped)": forest.predict(X_test),
    }

    print(f"Test rows: {len(y_test)}   (same split as train_models.py)\n")
    print(f"{'model':<30}{'MAE (kcal)':>12}{'R2':>9}")
    print("-" * 51)
    for name, prediction in predictions.items():
        print(f"{name:<30}{mean_absolute_error(y_true, prediction):>12.1f}{r2_score(y_true, prediction):>9.3f}")

    recorded = json.loads(METRICS_FILE.read_text(encoding="utf-8"))["test_metrics"]["random_forest"]
    mae_now = round(mean_absolute_error(y_true, predictions["Random Forest (shipped)"]), 1)
    print(f"\nRecorded in model_metrics.json: MAE {recorded['mae_kcal']}, R2 {recorded['r2']}"
          f"  ->  {'matches' if mae_now == recorded['mae_kcal'] else 'DIFFERS from'} the committed model")

    print(f"\nChange in MAE, forest minus rival, {BOOTSTRAP_RESAMPLES:,} bootstrap resamples of the test rows")
    print("(negative = the forest is better; the interval is 95%)\n")
    forest_prediction = predictions["Random Forest (shipped)"]
    for rival in ("Mifflin-St Jeor x 1.375", "linear regression (fitted)"):
        low, high, share = bootstrap_mae_difference(y_true, forest_prediction, predictions[rival])
        verdict = "interval excludes zero" if high < 0 or low > 0 else "interval includes zero"
        print(f"  vs {rival:<28} [{low:+6.1f}, {high:+6.1f}] kcal   "
              f"forest lower in {share:.0%} of resamples   ({verdict})")


if __name__ == "__main__":
    sys.exit(main())
