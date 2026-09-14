import random

from src.macro_predictor import MacroPredictor
from src.meal_recommender import MealRecommender


def test_macro_predictor_loads_and_predicts():
    predictor = MacroPredictor()
    assert predictor.is_available

    result = predictor.predict(height_cm=180, weight_kg=75, calories=2600)
    for key in ("protein_g", "carbs_g", "fat_g", "sugar_g"):
        assert key in result
        assert result[key] >= 0


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
    recommender = MealRecommender()
    plans = recommender.recommend_meal_plans(
        protein_target=150, fat_target=80, carbs_target=300, plan_count=1
    )
    names = [meal["food_name"] for meal in plans[0]["meals"]]
    assert len(names) == len(set(names))


def test_meal_recommender_plans_are_not_all_identical():
    # The candidate pick is random, so seed it to keep this deterministic
    # rather than relying on three draws happening to differ.
    random.seed(12345)
    recommender = MealRecommender()

    plans = recommender.recommend_meal_plans(
        protein_target=120, fat_target=70, carbs_target=250, plan_count=3
    )
    assert len(plans) == 3
    assert not all(plan == plans[0] for plan in plans)
