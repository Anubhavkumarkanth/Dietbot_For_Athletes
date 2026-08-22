from src.calculations import (
    bmi_category,
    calculate_bmi,
    calculate_bmr,
    calculate_daily_calorie_target,
    calculate_workout_calories,
)


def test_calculate_bmi():
    assert calculate_bmi(180, 81) == 25.0


def test_bmi_category_boundaries():
    assert bmi_category(17) == "Underweight"
    assert bmi_category(22) == "Normal weight"
    assert bmi_category(27) == "Overweight"
    assert bmi_category(32) == "Obese"


def test_calculate_bmr_male_vs_female():
    male_bmr = calculate_bmr(180, 80, 25, "Male")
    female_bmr = calculate_bmr(180, 80, 25, "Female")
    # Same height/weight/age - the Mifflin-St Jeor offset should make the
    # male figure exactly 166 higher than the female one (+5 vs -161).
    assert round(male_bmr - female_bmr) == 166


def test_calculate_workout_calories_known_sport():
    calories = calculate_workout_calories(70, 60, "Basketball")
    assert round(calories) == 455  # 6.5 MET * 70kg * 1 hour


def test_calculate_workout_calories_unknown_sport_uses_default():
    calories = calculate_workout_calories(70, 60, "Not A Real Sport")
    assert calories > 0


def test_daily_calorie_target_reflects_goal():
    loss = calculate_daily_calorie_target(180, 80, 25, "Male", "Sedentary", "Weight Loss")
    gain = calculate_daily_calorie_target(180, 80, 25, "Male", "Sedentary", "Weight Gain")
    assert gain > loss
    assert gain - loss == 1000  # -500 vs +500 adjustment
