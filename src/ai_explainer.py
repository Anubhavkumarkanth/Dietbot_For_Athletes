"""The only part of the app that calls an external AI API.

Everything else - BMI, BMR/TDEE, the macro prediction, the meal matching -
works without this module. All it does is turn numbers the app already
computed into a short, plain-language summary. If OPENAI_API_KEY isn't set,
or the request fails or times out, it falls back to a templated summary
instead of breaking the page.
"""

import os

from openai import OpenAI, OpenAIError

_client = OpenAI() if os.getenv("OPENAI_API_KEY") else None

_MODEL = "gpt-4o-mini"


def is_available() -> bool:
    return _client is not None


def explain_plan(profile: dict, macro_targets: dict, meal_plan: dict) -> str:
    """profile: {age, gender, sport, duration, goal}
    macro_targets: {protein_g, carbs_g, fat_g, sugar_g}
    meal_plan: {meals: [...], totals: {...}} - one plan from MealRecommender
    """
    if not is_available():
        return _fallback_explanation(macro_targets, meal_plan)

    food_list = ", ".join(item["food_name"] for item in meal_plan["meals"]) or "the foods below"
    prompt = (
        "You are a sports nutrition assistant. In 3-4 short, encouraging sentences, "
        "explain this daily plan to the athlete in plain language - no bullet points.\n"
        f"Athlete: {profile['age']} year old {profile['gender']}, plays {profile['sport']} "
        f"for {profile['duration']} minutes, goal is {profile['goal']}.\n"
        f"Daily targets: {macro_targets['protein_g']}g protein, {macro_targets['carbs_g']}g carbs, "
        f"{macro_targets['fat_g']}g fat.\n"
        f"Suggested foods: {food_list}."
    )

    try:
        response = _client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            timeout=10,
        )
        return response.choices[0].message.content.strip()
    except OpenAIError:
        return _fallback_explanation(macro_targets, meal_plan)


def _fallback_explanation(macro_targets: dict, meal_plan: dict) -> str:
    food_list = ", ".join(item["food_name"] for item in meal_plan["meals"]) or "a mix of foods"
    return (
        f"This plan targets {macro_targets['protein_g']}g protein, "
        f"{macro_targets['carbs_g']}g carbs and {macro_targets['fat_g']}g fat for the day, "
        f"built around {food_list}."
    )
