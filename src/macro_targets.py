"""Daily macro targets from a calorie target.

This was a Random Forest until its training labels turned out to be a fixed
multiple of the calorie column, so it was only rediscovering a formula - and it
could not extrapolate past 2,953 kcal. Applying the split directly fixes that.

The shares are that dataset's implied split normalised to 100% (it came to 103%,
which is its own tell). Fixed, so it does not vary by sport or goal - the data
had nothing to say about either.
"""

PROTEIN_ENERGY_SHARE = 0.236
CARB_ENERGY_SHARE = 0.516
FAT_ENERGY_SHARE = 0.248

# Sugar is a component of the carbohydrate total, not an addition to it, so this
# is reported as a suggested ceiling rather than counted toward the 100% above.
SUGAR_ENERGY_SHARE = 0.107

# Atwater factors.
KCAL_PER_GRAM_PROTEIN = 4
KCAL_PER_GRAM_CARB = 4
KCAL_PER_GRAM_FAT = 9


def calculate_macro_targets(daily_calories: float) -> dict:
    """Grams of protein, carbs and fat for a daily calorie target.

    Sugar is reported as a suggested ceiling, not added on top.
    """
    if daily_calories <= 0:
        raise ValueError(f"daily_calories must be positive, got {daily_calories}")

    return {
        "protein_g": round(daily_calories * PROTEIN_ENERGY_SHARE / KCAL_PER_GRAM_PROTEIN, 1),
        "carbs_g": round(daily_calories * CARB_ENERGY_SHARE / KCAL_PER_GRAM_CARB, 1),
        "fat_g": round(daily_calories * FAT_ENERGY_SHARE / KCAL_PER_GRAM_FAT, 1),
        "sugar_g": round(daily_calories * SUGAR_ENERGY_SHARE / KCAL_PER_GRAM_CARB, 1),
    }
