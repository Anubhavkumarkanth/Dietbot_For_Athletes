import json
import random
from pathlib import Path

import pytest

from src.calorie_predictor import FEATURE_COLUMNS, CaloriePredictor
from src.macro_targets import (
    CARB_ENERGY_SHARE,
    FAT_ENERGY_SHARE,
    PROTEIN_ENERGY_SHARE,
    calculate_macro_targets,
)
from src.meal_recommender import MealRecommender

METRICS_PATH = Path(__file__).resolve().parent.parent / "models" / "model_metrics.json"


def test_calorie_model_loads():
    assert CaloriePredictor().is_available


def test_calorie_prediction_type_and_plausible_range():
    prediction = CaloriePredictor().predict(
        age_years=30, is_male=True, height_cm=178, weight_kg=75, bmi=23.7
    )
    assert isinstance(prediction, float)
    # Trained on adults recorded eating 801-4996 kcal, so a prediction far
    # outside that band would mean something is wrong with the inputs or model.
    assert 800 <= prediction <= 5000


def test_calorie_prediction_responds_to_inputs():
    """A larger young man should be predicted to eat more than a smaller older woman.

    This is the direction the NHANES data supports (sex is the single largest
    feature importance), so a model that ignored its inputs would fail here.
    """
    predictor = CaloriePredictor()
    larger = predictor.predict(
        age_years=25, is_male=True, height_cm=188, weight_kg=90, bmi=25.5
    )
    smaller = predictor.predict(
        age_years=70, is_male=False, height_cm=155, weight_kg=52, bmi=21.6
    )
    assert larger > smaller


def test_calorie_predictions_are_not_constant():
    """Regression test for the flatline failure of the previous model.

    The old RandomForest was trained on 99 synthetic rows and could not
    extrapolate, returning byte-identical output for every input above its
    2,953 kcal ceiling. Any replacement must actually vary with its inputs.
    """
    predictor = CaloriePredictor()
    predictions = {
        predictor.predict(
            age_years=age, is_male=male, height_cm=height, weight_kg=weight, bmi=bmi
        )
        for age, male, height, weight, bmi in [
            (20, True, 185, 85, 24.8),
            (35, True, 175, 78, 25.5),
            (45, False, 168, 70, 24.8),
            (60, False, 160, 60, 23.4),
            (75, True, 170, 68, 23.5),
        ]
    }
    assert len(predictions) > 1, "model returned the same value for every profile"


def test_calorie_predictor_rejects_invalid_measurements():
    predictor = CaloriePredictor()
    with pytest.raises(ValueError):
        predictor.predict(age_years=0, is_male=True, height_cm=178, weight_kg=75, bmi=23.7)
    with pytest.raises(ValueError):
        predictor.predict(age_years=30, is_male=True, height_cm=178, weight_kg=-5, bmi=23.7)


def test_recorded_metrics_show_the_model_beats_the_formula():
    """The model only justifies its existence if it beats Mifflin-St Jeor.

    These numbers are written by scripts/train_models.py from a held-out test
    split. If a retrain ever stops beating the formula, this fails rather than
    letting the README keep claiming it does.
    """
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    forest = metrics["test_metrics"]["random_forest"]
    formula = metrics["test_metrics"]["mifflin_st_jeor"]

    assert forest["r2"] > formula["r2"]
    assert forest["mae_kcal"] < formula["mae_kcal"]
    assert metrics["random_forest_beats_formula_baseline"] is True
    assert metrics["features"] == FEATURE_COLUMNS
    assert metrics["dataset_rows"] > 4000


def test_macro_shares_sum_to_one():
    total = PROTEIN_ENERGY_SHARE + CARB_ENERGY_SHARE + FAT_ENERGY_SHARE
    assert total == pytest.approx(1.0, abs=1e-9)


def test_macro_targets_reconcile_with_the_calorie_target():
    calories = 3000
    macros = calculate_macro_targets(calories)
    from_macros = macros["protein_g"] * 4 + macros["carbs_g"] * 4 + macros["fat_g"] * 9
    assert from_macros == pytest.approx(calories, rel=0.01)


def test_macro_targets_scale_above_the_old_training_ceiling():
    at_3000 = calculate_macro_targets(3000)
    at_6000 = calculate_macro_targets(6000)
    for key in ("protein_g", "carbs_g", "fat_g", "sugar_g"):
        assert at_3000[key] < at_6000[key], f"{key} stopped scaling"
    assert at_6000["protein_g"] == pytest.approx(at_3000["protein_g"] * 2, rel=1e-6)


def test_macro_targets_rejects_non_positive_calories():
    with pytest.raises(ValueError):
        calculate_macro_targets(0)


def test_meal_recommender_loads_and_builds_plans():
    recommender = MealRecommender()
    assert recommender.is_available

    plans = recommender.recommend_meal_plans(
        protein_target=120, fat_target=70, carbs_target=250, plan_count=2
    )
    assert len(plans) == 2
    for plan in plans:
        assert len(plan["meals"]) > 0
        assert set(plan["totals"]) == {"protein", "fat", "carbs"}


def test_meal_recommender_plans_do_not_repeat_foods_within_a_plan():
    plans = MealRecommender().recommend_meal_plans(
        protein_target=150, fat_target=80, carbs_target=300, plan_count=1
    )
    names = [meal["food_name"] for meal in plans[0]["meals"]]
    assert len(names) == len(set(names))


def test_meal_recommender_plans_are_not_all_identical():
    # The candidate pick is random, so seed it to keep this deterministic
    # rather than relying on three draws happening to differ.
    random.seed(12345)
    plans = MealRecommender().recommend_meal_plans(
        protein_target=120, fat_target=70, carbs_target=250, plan_count=3
    )
    assert len(plans) == 3
    assert not all(plan == plans[0] for plan in plans)
