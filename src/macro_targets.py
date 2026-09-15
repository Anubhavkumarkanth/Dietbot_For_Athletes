"""Daily macro targets from a daily calorie target.

WHY THIS IS NOT A MODEL ANYMORE
-------------------------------
This used to load a 900 KB RandomForestRegressor trained on
``data/macro_targets.xlsx`` (99 usable rows, features: height, weight,
calories). Inspecting that dataset shows every target is a fixed multiple of
the calorie column:

    grams per calorie      mean       std        corr with calories
    protein                0.060974   0.000141   0.999940
    carbs                  0.133316   0.000134   0.999988
    fat                    0.028419   0.000125   0.999769
    sugar                  0.026695   0.000131   0.999720

A standard deviation in the fourth decimal place and a correlation of 0.9999
mean the labels were generated from a formula, not observed. Height and weight
carry almost no signal at all once calories are known (|r| < 0.12 against the
protein ratio), so the forest was learning ``y = k * x`` with two noise inputs.

Keeping it caused a real bug. A RandomForest cannot extrapolate: the training
data topped out at 2,953 kcal, so every athlete above roughly 2,900 kcal/day
received byte-identical macros. A 3,000 kcal cyclist and a 6,000 kcal swimmer
got the same numbers - which is precisely the population this app exists for.

Replacing the model with the split it had memorised fixes that, because a ratio
scales at any calorie level. It also removes a dependency on three pickle files
and makes the logic inspectable.

WHERE THESE NUMBERS COME FROM
-----------------------------
The dataset's implied split is 24.4% protein / 53.3% carbohydrate / 25.6% fat
of daily energy - which totals 103.3%, another sign it was synthetic rather
than measured. The shares below are those same proportions normalised to sum to
100%, so the macros now reconcile with the calorie target they came from.

KNOWN LIMITATION
----------------
The split is fixed. It does not change with training load or goal, because the
original dataset contained no information about either - it only ever saw
height, weight and calories. Goal affects the calorie target (see
``calculations.calculate_daily_calorie_target``), and the macros scale with it,
but the ratio between them stays constant. Varying the split by sport or goal
would need nutrition data this project does not have.
"""

# Share of daily energy from each macronutrient. These sum to 1.0.
PROTEIN_ENERGY_SHARE = 0.236
CARB_ENERGY_SHARE = 0.516
FAT_ENERGY_SHARE = 0.248

# Sugar is a component of the carbohydrate total, not an addition to it, so this
# is reported as a suggested ceiling rather than counted toward the 100% above.
SUGAR_ENERGY_SHARE = 0.107

# Atwater factors: metabolisable energy per gram.
KCAL_PER_GRAM_PROTEIN = 4
KCAL_PER_GRAM_CARB = 4
KCAL_PER_GRAM_FAT = 9


def calculate_macro_targets(daily_calories: float) -> dict:
    """Convert a daily calorie target into grams of each macronutrient.

    Args:
        daily_calories: Total daily calorie target, as produced by
            ``calculations.calculate_daily_calorie_target``.

    Returns:
        Grams per day of protein, carbs, fat, and a suggested sugar ceiling.

    Raises:
        ValueError: If ``daily_calories`` is not positive.
    """
    if daily_calories <= 0:
        raise ValueError(f"daily_calories must be positive, got {daily_calories}")

    return {
        "protein_g": round(daily_calories * PROTEIN_ENERGY_SHARE / KCAL_PER_GRAM_PROTEIN, 1),
        "carbs_g": round(daily_calories * CARB_ENERGY_SHARE / KCAL_PER_GRAM_CARB, 1),
        "fat_g": round(daily_calories * FAT_ENERGY_SHARE / KCAL_PER_GRAM_FAT, 1),
        "sugar_g": round(daily_calories * SUGAR_ENERGY_SHARE / KCAL_PER_GRAM_CARB, 1),
    }
