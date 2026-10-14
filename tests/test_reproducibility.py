"""The published numbers must be reproducible from what is committed.

test_models.py only reads models/model_metrics.json. The tests here recompute the
held-out metrics from data/nhanes_intake.csv and the committed model, using the
split and baseline code in scripts/train_models.py itself, and compare them with
what the JSON and the README claim. They fail if the data, the model file and
the recorded metrics ever drift apart, for example after a retrain that was not
committed together.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import train_models  # noqa: E402  (scripts/ is not a package)

RECORDED = json.loads((ROOT / "models" / "model_metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dataset():
    return pd.read_csv(train_models.INTAKE_CSV)


@pytest.fixture(scope="module")
def held_out(dataset):
    X, y = dataset[train_models.FEATURES], dataset[train_models.TARGET]
    X_train, _, X_test, y_train, _, y_test = train_models._split(X, y)
    return X_train, y_train, X_test, y_test


# ------------------------------------------------------------------ the metrics
def test_split_sizes_match_the_recorded_split(held_out):
    X_train, _, X_test, _ = held_out
    assert len(X_train) == RECORDED["split"]["train"]
    assert len(X_test) == RECORDED["split"]["test"]


def test_committed_model_reproduces_the_recorded_test_metrics(held_out):
    _, _, X_test, y_test = held_out
    model = joblib.load(train_models.CALORIE_MODEL)
    predicted = model.predict(X_test)

    recorded = RECORDED["test_metrics"]["random_forest"]
    assert round(mean_absolute_error(y_test, predicted), 1) == recorded["mae_kcal"]
    assert r2_score(y_test, predicted) == pytest.approx(recorded["r2"], abs=5e-4)


def test_baselines_reproduce_the_recorded_numbers(held_out):
    _, y_train, X_test, y_test = held_out

    mean_prediction = np.full(len(y_test), y_train.mean())
    formula_prediction = train_models.mifflin_st_jeor(X_test)

    mean_recorded = RECORDED["test_metrics"]["mean_baseline"]
    formula_recorded = RECORDED["test_metrics"]["mifflin_st_jeor"]
    assert round(mean_absolute_error(y_test, mean_prediction), 1) == mean_recorded["mae_kcal"]
    assert round(mean_absolute_error(y_test, formula_prediction), 1) == formula_recorded["mae_kcal"]
    assert r2_score(y_test, formula_prediction) == pytest.approx(formula_recorded["r2"], abs=5e-4)


# ------------------------------------------------------------------ the data
def test_dataset_has_the_documented_shape(dataset):
    assert len(dataset) == RECORDED["dataset_rows"] == 4624
    assert list(dataset.columns) == [
        "age_years", "height_cm", "weight_kg", "bmi", "energy_kcal",
        "protein_g", "carbs_g", "fat_g", "sugar_g", "SEX_MALE",
    ]


def test_dataset_is_clean(dataset):
    assert not dataset.isna().any().any()
    assert not dataset.duplicated().any()
    assert set(dataset["SEX_MALE"].unique()) == {0, 1}


def test_dataset_respects_the_filters_build_dataset_applies(dataset):
    assert dataset["age_years"].between(18, 80).all()
    assert dataset["energy_kcal"].between(800, 5000).all()
    assert (dataset[["protein_g", "carbs_g", "fat_g"]] > 0).all().all()


def test_bmi_column_agrees_with_height_and_weight(dataset):
    computed = dataset["weight_kg"] / (dataset["height_cm"] / 100) ** 2
    assert (computed - dataset["bmi"]).abs().max() < 0.1


def test_recorded_calories_agree_with_the_macros(dataset):
    """4/4/9 kcal per gram of protein/carbohydrate/fat should land near the recorded energy.

    Alcohol and rounding make an exact match impossible, so this checks the
    typical row, not every row.
    """
    from_macros = 4 * dataset["protein_g"] + 4 * dataset["carbs_g"] + 9 * dataset["fat_g"]
    ratio = (from_macros / dataset["energy_kcal"]).median()
    assert 0.95 < ratio < 1.05
