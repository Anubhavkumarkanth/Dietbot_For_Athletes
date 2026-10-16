"""Smoke tests for the Streamlit app, run headless with Streamlit's AppTest.

They check that the page renders and that pressing the button produces targets
and meal plans, which is the path a user takes. They do not check how it looks.

app.py calls load_dotenv() and, if the settings exist, talks to OpenAI and to a
database. A test must do neither, so the fixture removes those settings and
stops load_dotenv from putting them back from a developer's .env.
"""

import os
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import storage
from src.calculations import calculate_daily_calorie_target

APP = str(Path(__file__).resolve().parent.parent / "app.py")

# The sidebar defaults: height 170 cm, weight 70 kg, age 25, and the first
# option of each selectbox.
DEFAULTS = dict(height_cm=170, weight_kg=70, age=25, gender="Male", activity_level="Sedentary")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DIETBOT_DB_URL", raising=False)

    from src import ai_explainer
    monkeypatch.setattr(ai_explainer, "_client", None)

    return AppTest.from_file(APP, default_timeout=60).run()


def metrics(at):
    return {metric.label: metric.value for metric in at.metric}


def press_get_recommendations(at):
    at.sidebar.button[0].click()
    return at.run()


def set_details(at, height=170, weight=70, age=25, gender="Male", activity_level="Sedentary", goal="Weight Loss"):
    """Fill in the sidebar: three number inputs, then the gender, sport, activity and goal selectboxes."""
    at.sidebar.number_input[0].set_value(height)
    at.sidebar.number_input[1].set_value(weight)
    at.sidebar.number_input[2].set_value(age)
    at.sidebar.selectbox[0].select(gender)
    at.sidebar.selectbox[2].select(activity_level)
    at.sidebar.selectbox[3].select(goal)


def test_the_page_renders_and_asks_for_details(app):
    assert not app.exception
    assert any("Fill in your details" in info.value for info in app.info)
    assert not app.expander, "no plans should be shown before the button is pressed"


def test_pressing_the_button_shows_targets_and_three_meal_plans(app):
    at = press_get_recommendations(app)
    assert not at.exception

    shown = metrics(at)
    expected_target = calculate_daily_calorie_target(
        DEFAULTS["height_cm"], DEFAULTS["weight_kg"], DEFAULTS["age"],
        DEFAULTS["gender"], DEFAULTS["activity_level"], "Weight Loss",
    )
    assert shown["Daily calorie target"] == f"{expected_target} kcal"
    assert {"Protein", "Carbs", "Fat", "Sugar (max)", "Typical intake for this profile"} <= set(shown)

    assert len(at.expander) == 3
    assert "Plan summary" in [subheader.value for subheader in at.subheader]


def test_changing_the_goal_changes_the_calorie_target(app):
    app.sidebar.selectbox[3].select("Weight Gain")
    at = press_get_recommendations(app)
    assert not at.exception

    expected_target = calculate_daily_calorie_target(
        DEFAULTS["height_cm"], DEFAULTS["weight_kg"], DEFAULTS["age"],
        DEFAULTS["gender"], DEFAULTS["activity_level"], "Weight Gain",
    )
    assert metrics(at)["Daily calorie target"] == f"{expected_target} kcal"


def test_without_a_database_no_history_is_offered(app):
    at = press_get_recommendations(app)
    assert not at.exception
    assert "Past plans" not in [subheader.value for subheader in at.subheader]
    assert not any("saved as plan" in caption.value.lower() for caption in at.caption)


def test_a_target_below_the_advised_minimum_shows_a_warning_and_keeps_the_number(app):
    set_details(app, height=150, weight=45, age=30, gender="Female")  # the formula gives 792 kcal
    at = press_get_recommendations(app)
    assert not at.exception
    assert metrics(at)["Daily calorie target"] == "792 kcal"
    assert any("below 1,200 kcal" in warning.value for warning in at.warning)
    assert len(at.expander) == 3, "the plan is still built: the warning explains the number, it does not withhold it"


def test_the_man_in_the_default_profile_is_warned_about_a_target_just_under_1500(app):
    at = press_get_recommendations(app)  # 170 cm, 70 kg, 25 years, sedentary, weight loss: 1,471 kcal
    assert not at.exception
    assert any("below 1,500 kcal" in warning.value for warning in at.warning)


def test_a_target_at_or_above_the_minimum_shows_no_warning(app):
    set_details(app, goal="Weight Gain")
    at = press_get_recommendations(app)
    assert not at.exception
    assert not at.warning


def test_a_target_that_is_not_positive_is_explained_instead_of_crashing(app):
    set_details(app, height=100, weight=30, age=70, gender="Female")  # the formula gives -3 kcal
    at = press_get_recommendations(app)
    assert not at.exception, "this used to raise ValueError: daily_calories must be positive, got -3"
    assert any("cannot be turned into a plan" in error.value for error in at.error)
    assert not at.expander, "no meal plans without a usable target"


def test_food_values_are_labelled_per_100_g(app):
    at = press_get_recommendations(app)
    assert any("per 100 g" in caption.value for caption in at.caption)


def test_sugar_is_labelled_as_a_maximum_inside_the_carbs(app):
    at = press_get_recommendations(app)
    assert "Sugar (max)" in metrics(at)
    assert any("part of the carbohydrate total" in caption.value for caption in at.caption)


def test_a_saved_plan_says_which_meal_plan_was_saved(monkeypatch):
    """Needs the _test database in DIETBOT_TEST_DB_URL (the one the persistence tests use); skipped without it."""
    url = os.environ.get("DIETBOT_TEST_DB_URL")
    if not url:
        pytest.skip("DIETBOT_TEST_DB_URL is not set")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DIETBOT_DB_URL", url)
    if not storage.is_available():
        pytest.skip("the test database is unreachable")

    from src import ai_explainer
    monkeypatch.setattr(ai_explainer, "_client", None)

    at = AppTest.from_file(APP, default_timeout=60).run()
    at = press_get_recommendations(at)
    saved = [m.group(1) for c in at.caption if (m := re.fullmatch(r"Meal plan 1 saved as plan #(\d+)\.", c.value))]
    try:
        assert len(saved) == 1, [c.value for c in at.caption]
    finally:
        for plan_id in saved:  # leave the test database as it was found
            with storage._connect() as connection:
                with connection:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            "DELETE FROM profiles WHERE profile_id = (SELECT profile_id FROM plans WHERE plan_id = %s)",
                            (int(plan_id),),
                        )
