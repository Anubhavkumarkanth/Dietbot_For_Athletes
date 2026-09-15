"""Trains the models and writes them to models/.

Two models are produced:

  1. calorie_intake_rf.pkl  - RandomForestRegressor predicting daily calorie
     intake from age, sex, height, weight and BMI, trained on NHANES.
  2. nearest_neighbors_model.pkl - NearestNeighbors over the food table, used to
     assemble meal plans against macro targets.

Both artifacts are committed, so you do not need to run this to use the app.
Run it if the data changes or you want to reproduce the models yourself:

    python scripts/train_models.py            # train and save both
    python scripts/train_models.py --rejected # reproduce the rejected designs

WHY THE MODEL PREDICTS CALORIES AND NOT MACROS
----------------------------------------------
The original model predicted macro grams. Four designs were evaluated on real
NHANES data before settling on this one, and three of them failed to beat a
trivial baseline. `--rejected` reproduces them. In short:

  macro grams from body + calories   RF R2 0.584  vs  0.585 for calories x a
                                     fixed ratio. The forest ties a one-line
                                     formula, so it earns nothing.
  macro grams from body alone        RF R2 0.043. Beats the mean, but that is
                                     not predictive power in any useful sense.
  macro composition (% of energy)    RF R2 -0.057, i.e. worse than predicting
                                     the training mean for everyone.
  calorie intake from body           RF R2 0.130 vs 0.071 for Mifflin-St Jeor,
                                     MAE 633 vs 651 kcal. The only design that
                                     beats its incumbent on both metrics.

The conclusion is about the relationship, not the dataset: once you know a
person's calorie intake, their macro grams follow the population ratio closely,
and body measurements explain almost none of what is left. Macro targets are
therefore computed directly in src/macro_targets.py, and the RandomForest does
the one job it measurably does better than the formula it replaces.

HONEST LIMITS OF THE SHIPPED MODEL
----------------------------------
R2 of 0.130 is low. A single 24-hour dietary recall is a noisy measurement of a
person's habitual intake (the test set's own standard deviation is about 855
kcal), so most of that variance is not predictable from body measurements by
any model. It predicts what people like you *typically eat*, which is not the
same thing as what you *should* eat. The app presents it as a reference
alongside the Mifflin-St Jeor target rather than as a recommendation.
"""

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

INTAKE_CSV = DATA_DIR / "nhanes_intake.csv"
CALORIE_MODEL = MODELS_DIR / "calorie_intake_rf.pkl"
METRICS_FILE = MODELS_DIR / "model_metrics.json"

# Order matters: src/calorie_predictor.py builds its input frame with these
# names in this order.
FEATURES = ["age_years", "SEX_MALE", "height_cm", "weight_kg", "bmi"]
TARGET = "energy_kcal"

FOOD_COLUMNS = {
    "Food_name": "food_name",
    "Protein(g)": "protein",
    "Total lipid (fat)(g)": "fat",
    "Carbohydrate, by difference(g)": "carbs",
}

RANDOM_STATE = 42


def _load_intake() -> pd.DataFrame:
    if not INTAKE_CSV.exists():
        raise SystemExit(
            f"{INTAKE_CSV} is missing. Run: python scripts/build_dataset.py"
        )
    return pd.read_csv(INTAKE_CSV)


def _split(X, y):
    """60/20/20 train/validation/test.

    The validation split is used to choose min_samples_leaf; the test split is
    touched once, at the end, so the reported numbers are not tuned against.
    """
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def mifflin_st_jeor(frame: pd.DataFrame) -> np.ndarray:
    """The formula the app already uses, as a baseline to beat.

    BMR x 1.375 (lightly active), matching calculations.calculate_bmr.
    """
    base = (
        10 * frame["weight_kg"]
        + 6.25 * frame["height_cm"]
        - 5 * frame["age_years"]
    )
    bmr = np.where(frame["SEX_MALE"] == 1, base + 5, base - 161)
    return bmr * 1.375


def train_calorie_model() -> None:
    df = _load_intake()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, y)

    print(f"Dataset: {len(df)} adults from NHANES 2017-2018")
    print(f"Split:   train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")
    print(f"Target:  {TARGET}    Features: {FEATURES}\n")

    # Pick leaf size on validation, not on test.
    best_leaf, best_val = None, -np.inf
    for leaf in (1, 5, 10, 20, 50, 100, 150, 200, 300):
        candidate = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=leaf,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ).fit(X_train, y_train)
        score = r2_score(y_val, candidate.predict(X_val))
        print(f"  min_samples_leaf={leaf:<3} validation R2={score:.4f}")
        if score > best_val:
            best_leaf, best_val = leaf, score
    print(f"\nChosen min_samples_leaf={best_leaf} (validation R2={best_val:.4f})\n")

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=best_leaf,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(X_train, y_train)

    predictions = {
        "mean_baseline": np.full(len(y_test), y_train.mean()),
        "mifflin_st_jeor": mifflin_st_jeor(X_test),
        "random_forest": model.predict(X_test),
    }

    print(f"{'model':<20}{'MAE (kcal)':>12}{'R2':>9}")
    print("-" * 41)
    metrics = {}
    for name, prediction in predictions.items():
        mae = float(mean_absolute_error(y_test, prediction))
        r2 = float(r2_score(y_test, prediction))
        metrics[name] = {"mae_kcal": round(mae, 1), "r2": round(r2, 4)}
        print(f"{name:<20}{mae:>12.1f}{r2:>9.3f}")

    beats = (
        metrics["random_forest"]["r2"] > metrics["mifflin_st_jeor"]["r2"]
        and metrics["random_forest"]["mae_kcal"] < metrics["mifflin_st_jeor"]["mae_kcal"]
    )
    print(f"\nRandomForest beats the Mifflin-St Jeor baseline: {beats}")
    print(f"Test-set standard deviation: {y_test.std():.0f} kcal "
          f"(a 24h recall is a noisy measure of habitual intake)")

    record = {
        "generated_on": date.today().isoformat(),
        "data_source": "NHANES 2017-2018 (US CDC / NCHS, public domain)",
        "dataset_rows": int(len(df)),
        "features": FEATURES,
        "target": TARGET,
        "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "min_samples_leaf": int(best_leaf),
        "test_metrics": metrics,
        "random_forest_beats_formula_baseline": bool(beats),
        "test_target_std_kcal": round(float(y_test.std()), 1),
        "feature_importances": {
            feature: round(float(value), 4)
            for feature, value in zip(FEATURES, model.feature_importances_)
        },
    }

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, CALORIE_MODEL)
    METRICS_FILE.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nSaved {CALORIE_MODEL.name} and {METRICS_FILE.name}")


def train_food_matcher() -> None:
    food_data = pd.read_csv(DATA_DIR / "food_data.csv", encoding="latin1")
    foods = (
        food_data[list(FOOD_COLUMNS)]
        .dropna()
        .rename(columns=FOOD_COLUMNS)
        .reset_index(drop=True)
    )

    model = NearestNeighbors(n_neighbors=5, metric="euclidean")
    model.fit(foods[["protein", "fat", "carbs"]].to_numpy())

    joblib.dump(model, MODELS_DIR / "nearest_neighbors_model.pkl")
    print(f"Food matcher trained on {len(foods)} food items")


def show_rejected_designs() -> None:
    """Reproduce the three model designs that did not earn their place."""
    df = _load_intake()
    macro_targets = ["protein_g", "carbs_g", "fat_g", "sugar_g"]

    def run(title, features, targets, baseline, baseline_name):
        X_train, _, X_test, y_train, _, y_test = _split(df[features], targets)
        model = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1
        ).fit(X_train, y_train)
        rf_r2 = r2_score(y_test, model.predict(X_test), multioutput="uniform_average")
        base_r2 = r2_score(y_test, baseline(X_train, y_train, X_test),
                           multioutput="uniform_average")
        verdict = "BEATS baseline" if rf_r2 > base_r2 else "does NOT beat baseline"
        print(f"\n{title}")
        print(f"  RandomForest R2 = {rf_r2:+.3f}")
        print(f"  {baseline_name:<28} R2 = {base_r2:+.3f}   -> {verdict}")

    def mean_baseline(X_train, y_train, X_test):
        return np.tile(np.asarray(y_train.mean()), (len(X_test), 1))

    def ratio_baseline(X_train, y_train, X_test):
        ratios = (y_train.values / X_train[["energy_kcal"]].values).mean(axis=0)
        return X_test[["energy_kcal"]].values * ratios

    print("Rejected model designs, reproduced on the same NHANES data")
    print("=" * 60)

    run("1. Macro grams from body measurements + calories",
        FEATURES + ["energy_kcal"], df[macro_targets], ratio_baseline,
        "calories x fixed ratio")

    run("2. Macro grams from body measurements alone",
        FEATURES, df[macro_targets], mean_baseline, "predict the training mean")

    composition = pd.DataFrame({
        "protein_g": df.protein_g * 4 / df.energy_kcal,
        "carbs_g": df.carbs_g * 4 / df.energy_kcal,
        "fat_g": df.fat_g * 9 / df.energy_kcal,
        "sugar_g": df.sugar_g * 4 / df.energy_kcal,
    })
    run("3. Macro composition as a share of energy",
        FEATURES, composition, mean_baseline, "predict the training mean")

    print("\nOnly the calorie-intake design beat its incumbent, so it is the one"
          "\nthat ships. Macro targets are computed in src/macro_targets.py.")


if __name__ == "__main__":
    if "--rejected" in sys.argv:
        show_rejected_designs()
    else:
        MODELS_DIR.mkdir(exist_ok=True)
        train_calorie_model()
        print()
        train_food_matcher()
        print("\nDone. Models written to", MODELS_DIR)
