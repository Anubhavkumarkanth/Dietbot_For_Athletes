"""Smoke tests for the Streamlit app, run headless with Streamlit's AppTest.

They check that the page renders and that pressing the button produces targets
and meal plans, which is the path a user takes. They do not check how it looks.

app.py calls load_dotenv() and, if the settings exist, talks to OpenAI and to a
database. A test must do neither, so the fixture removes those settings and
stops load_dotenv from putting them back from a developer's .env.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

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
    assert {"Protein", "Carbs", "Fat", "Sugar", "Typical intake for this profile"} <= set(shown)

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
    assert not any("Saved as plan" in caption.value for caption in at.caption)
