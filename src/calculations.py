"""Deterministic nutrition and activity math.

Nothing in this module touches a model or an API - it's BMI, BMR/TDEE and
MET-based calorie burn, all standard formulas. Keeping it separate from
macro_predictor.py and meal_recommender.py makes it obvious which parts of
the app need the trained models to work and which parts don't.
"""

MET_VALUES = {
    "Basketball": 6.5,
    "Football": 9.0,
    "Soccer": 7.0,
    "Swimming": 8.0,
    "Running": 9.8,
    "Cycling": 7.5,
    "Tennis": 7.3,
    "Weightlifting": 3.0,
    "Volleyball": 4.0,
    "Gymnastics": 3.8,
    "Boxing": 9.0,
    "Wrestling": 6.0,
    "Hockey": 8.5,
    "Badminton": 5.5,
    "Table Tennis": 4.0,
    "Track and Field": 9.0,
    "Baseball": 5.0,
    "Softball": 5.0,
    "Golf": 4.8,
    "Skiing": 7.0,
    "Rowing": 6.0,
    "Martial Arts": 10.0,
    "Yoga": 2.5,
    "CrossFit": 9.5,
}

ACTIVITY_MULTIPLIERS = {
    "Sedentary": 1.2,
    "Lightly Active": 1.375,
    "Moderately Active": 1.55,
    "Very Active": 1.725,
    "Extra Active": 1.9,
}

# Rough daily calorie surplus/deficit applied on top of TDEE for each goal.
GOAL_CALORIE_ADJUSTMENT = {
    "Weight Loss": -500,
    "Muscle Gain": 300,
    "Weight Gain": 500,
}

BMI_CATEGORIES = [
    (18.5, "Underweight"),
    (25.0, "Normal weight"),
    (30.0, "Overweight"),
]


def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)


def bmi_category(bmi: float) -> str:
    for threshold, label in BMI_CATEGORIES:
        if bmi < threshold:
            return label
    return "Obese"


def calculate_bmr(height_cm: float, weight_kg: float, age: int, gender: str) -> float:
    """Mifflin-St Jeor equation - the standard BMR formula used by most
    modern calorie calculators (more accurate than the older Harris-Benedict
    version)."""
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return base + 5 if gender == "Male" else base - 161


def calculate_workout_calories(weight_kg: float, duration_minutes: float, sport: str) -> float:
    """Calories burned during a single training session, not a daily total."""
    met = MET_VALUES.get(sport, 5.0)
    return met * weight_kg * (duration_minutes / 60)


def calculate_daily_calorie_target(
    height_cm: float,
    weight_kg: float,
    age: int,
    gender: str,
    activity_level: str,
    goal: str,
) -> int:
    """Total daily calorie target (BMR x activity factor, adjusted for goal).

    This is what should feed the macro-prediction model - not the single
    workout's calorie burn, which is a much smaller number and was being
    used for that purpose in the original prototype.
    """
    bmr = calculate_bmr(height_cm, weight_kg, age, gender)
    tdee = bmr * ACTIVITY_MULTIPLIERS.get(activity_level, 1.375)
    adjustment = GOAL_CALORIE_ADJUSTMENT.get(goal, 0)
    return round(tdee + adjustment)
