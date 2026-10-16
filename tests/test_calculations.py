from src.calculations import (
    bmi_category,
    calculate_bmi,
    calculate_bmr,
    calculate_daily_calorie_target,
    calculate_workout_calories,
    low_calorie_warning,
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


def test_low_calorie_warning_starts_below_the_commonly_advised_minimums():
    assert low_calorie_warning(1199, "Female") is not None
    assert low_calorie_warning(1200, "Female") is None
    assert low_calorie_warning(1499, "Male") is not None
    assert low_calorie_warning(1500, "Male") is None


def test_low_calorie_warning_names_the_minimum_that_applies():
    assert "1,200" in low_calorie_warning(800, "Female")
    assert "1,500" in low_calorie_warning(800, "Male")


def test_an_unrecognised_gender_gets_the_lower_minimum():
    assert low_calorie_warning(1199, "Other") is not None
    assert low_calorie_warning(1300, "Other") is None


def test_the_warning_does_not_change_the_target():
    # A 150 cm, 45 kg, 30-year-old woman on a weight-loss goal: the formula gives 792 kcal and still does.
    target = calculate_daily_calorie_target(150, 45, 30, "Female", "Sedentary", "Weight Loss")
    assert target == 792
    assert low_calorie_warning(target, "Female") is not None
    assert calculate_daily_calorie_target(150, 45, 30, "Female", "Sedentary", "Weight Loss") == 792
