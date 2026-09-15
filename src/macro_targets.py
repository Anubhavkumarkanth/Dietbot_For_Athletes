"""Daily macro targets from a daily calorie target.

This used to be a Random Forest. Its training targets turned out to be a fixed
multiple of the calorie column (grams per calorie varied by ~0.0001), so the
model was only rediscovering a formula, and being tree-based it returned
identical macros for anything above its 2,953 kcal training ceiling. Applying
the split directly fixes that and removes three pickle files. See the README for
the numbers.

The shares below are that dataset's implied split, normalised to sum to 100% —
it originally totalled 103%, which is itself a sign it was generated.

The split is fixed: it does not vary by sport or goal, because the original data
contained no information about either. Goal changes the calorie target, and the
macros scale with it.
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
