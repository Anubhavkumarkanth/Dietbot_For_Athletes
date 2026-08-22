import streamlit as st

from src import ai_explainer
from src.calculations import (
    ACTIVITY_MULTIPLIERS,
    GOAL_CALORIE_ADJUSTMENT,
    MET_VALUES,
    bmi_category,
    calculate_bmi,
    calculate_daily_calorie_target,
    calculate_workout_calories,
)
from src.macro_predictor import MacroPredictor
from src.meal_recommender import MealRecommender

st.set_page_config(page_title="DietBot - Athlete Diet Recommendations", page_icon="🥗", layout="wide")

macro_predictor = MacroPredictor()
meal_recommender = MealRecommender()

st.sidebar.title("Your Details")
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
sport = st.sidebar.selectbox("Sport", sorted(MET_VALUES.keys()))
duration = st.sidebar.slider("Duration of activity (minutes)", min_value=10, max_value=180, value=60)
activity_level = st.sidebar.selectbox("Activity level", list(ACTIVITY_MULTIPLIERS.keys()))
goal = st.sidebar.selectbox("Fitness goal", list(GOAL_CALORIE_ADJUSTMENT.keys()))

st.title("🥗 DietBot — Athlete Diet Recommendations")
st.caption("Personalized macro targets and meal ideas based on your stats and training load.")

if not macro_predictor.is_available:
    st.error(
        "The macro prediction model couldn't be loaded. Make sure the files in "
        "`models/` are present, or run `python scripts/train_models.py` to regenerate them."
    )
    st.stop()

if st.sidebar.button("Get recommendations", type="primary"):
    bmi = calculate_bmi(height, weight)
    workout_calories = calculate_workout_calories(weight, duration, sport)
    daily_calories = calculate_daily_calorie_target(height, weight, age, gender, activity_level, goal)

    col1, col2, col3 = st.columns(3)
    col1.metric("BMI", f"{bmi} ({bmi_category(bmi)})")
    col2.metric("Calories burned this session", f"{workout_calories:.0f} kcal")
    col3.metric("Daily calorie target", f"{daily_calories} kcal")

    try:
        macro_targets = macro_predictor.predict(height, weight, daily_calories)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Daily macro targets")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Protein", f"{macro_targets['protein_g']} g")
    m2.metric("Carbs", f"{macro_targets['carbs_g']} g")
    m3.metric("Fat", f"{macro_targets['fat_g']} g")
    m4.metric("Sugar", f"{macro_targets['sugar_g']} g")

    st.subheader("🍽 Recommended meal plans")

    if not meal_recommender.is_available:
        st.warning("Meal recommendations are unavailable — food data or the matching model failed to load.")
        st.stop()

    plans = meal_recommender.recommend_meal_plans(
        macro_targets["protein_g"], macro_targets["fat_g"], macro_targets["carbs_g"]
    )

    for i, plan in enumerate(plans, start=1):
        with st.expander(f"Meal plan {i}", expanded=(i == 1)):
            for item in plan["meals"]:
                st.write(
                    f"- {item['food_name']} "
                    f"(Protein {item['protein']}g, Fat {item['fat']}g, Carbs {item['carbs']}g)"
                )
            totals = plan["totals"]
            st.caption(
                f"Totals: {totals['protein']}g protein · {totals['fat']}g fat · {totals['carbs']}g carbs"
            )

    st.subheader("Plan summary")
    if ai_explainer.is_available():
        with st.spinner("Writing a summary..."):
            explanation = ai_explainer.explain_plan(
                {"age": age, "gender": gender, "sport": sport, "duration": duration, "goal": goal},
                macro_targets,
                plans[0],
            )
        st.info(explanation)
    else:
        st.info(ai_explainer.explain_plan(
            {"age": age, "gender": gender, "sport": sport, "duration": duration, "goal": goal},
            macro_targets,
            plans[0],
        ))
        st.caption("Set OPENAI_API_KEY in your environment to get an AI-generated summary here instead.")
else:
    st.info("Fill in your details on the left and click **Get recommendations** to see your plan.")
