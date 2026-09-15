import streamlit as st
from dotenv import load_dotenv

# Without this, python-dotenv was a declared dependency that nothing used,
# and copying .env.example to .env had no effect.
load_dotenv()

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
from src.calorie_predictor import CaloriePredictor
from src.macro_targets import calculate_macro_targets
from src.meal_recommender import MealRecommender
from src import storage

st.set_page_config(page_title="DietBot - Athlete Diet Recommendations", page_icon="🥗", layout="wide")

calorie_predictor = CaloriePredictor()
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

if st.sidebar.button("Get recommendations", type="primary"):
    bmi = calculate_bmi(height, weight)
    workout_calories = calculate_workout_calories(weight, duration, sport)
    daily_calories = calculate_daily_calorie_target(height, weight, age, gender, activity_level, goal)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BMI", f"{bmi} ({bmi_category(bmi)})")
    col2.metric("Calories burned this session", f"{workout_calories:.0f} kcal")
    col3.metric("Daily calorie target", f"{daily_calories} kcal")

    # Reference figure from the trained model. Descriptive, not prescriptive:
    # it is what people with this profile were recorded eating, which is a
    # different question from what this user should eat. The plan below is
    # built from the target in col3, not from this number.
    # Bound before the try so the save step below can rely on it existing
    # whether or not the prediction succeeded.
    typical_intake = None

    if calorie_predictor.is_available:
        try:
            typical_intake = calorie_predictor.predict(
                age_years=age, is_male=(gender == "Male"),
                height_cm=height, weight_kg=weight, bmi=bmi,
            )
            col4.metric("Typical intake for this profile", f"{typical_intake:.0f} kcal")
        except (RuntimeError, ValueError):
            col4.metric("Typical intake for this profile", "unavailable")
        else:
            st.caption(
                "'Typical intake' is predicted by a Random Forest trained on NHANES "
                "2017-2018 (4,624 adults, US CDC, public domain). It describes what "
                "people with similar age, sex and body composition were recorded "
                "eating - not a recommendation. Your plan below uses the calorie "
                "target, not this figure."
            )

    # A direct calculation from the calorie target - see src/macro_targets.py
    # for why this replaced a trained model.
    macro_targets = calculate_macro_targets(daily_calories)

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

    if storage.is_available():
        try:
            saved_plan_id = storage.save_plan(
                profile={
                    "height_cm": height, "weight_kg": weight, "age_years": age,
                    "gender": gender, "sport": sport, "duration_min": duration,
                    "activity_level": activity_level, "goal": goal,
                },
                targets={"daily_calories": daily_calories, **macro_targets},
                meals=plans[0]["meals"],
                predicted_intake_kcal=typical_intake,
            )
            st.caption(f"Saved as plan #{saved_plan_id}.")
        except Exception as exc:
            # Saving is a convenience; failing to save must not lose the plan
            # the user is looking at.
            st.caption(f"Could not save this plan: {exc}")

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


# --- Past plans -------------------------------------------------------------
if storage.is_available():
    st.divider()
    st.subheader("Past plans")
    try:
        history = storage.recent_plans(limit=5)
    except Exception as exc:
        history = []
        st.caption(f"Could not load past plans: {exc}")

    if not history:
        st.caption("No saved plans yet.")
    else:
        for row in history:
            label = (
                f"#{row['plan_id']} · {row['created_at']:%d %b %Y %H:%M} · "
                f"{row['sport']} · {row['goal']} · {row['daily_calories']} kcal target"
            )
            with st.expander(label):
                st.write(
                    f"Targets: {row['protein_g']}g protein · "
                    f"{row['carbs_g']}g carbs · {row['fat_g']}g fat"
                )
                if row["predicted_intake_kcal"]:
                    st.caption(
                        f"Model estimate of typical intake at the time: "
                        f"{row['predicted_intake_kcal']} kcal"
                    )
                for item in storage.plan_items(row["plan_id"]):
                    st.write(
                        f"- {item['food_name']} "
                        f"(Protein {item['protein_g']}g, Fat {item['fat_g']}g, "
                        f"Carbs {item['carbs_g']}g)"
                    )
