"""Tests for the PostgreSQL persistence layer.

These need a live database. When DIETBOT_DB_URL is unset or unreachable they
skip rather than fail, because persistence is an optional feature - the app runs
without it, and so should the rest of the suite. Run them by setting
DIETBOT_DB_URL and applying sql/schema.sql first.
"""

import pytest

from src import storage

pytestmark = pytest.mark.skipif(
    not storage.is_available(),
    reason="DIETBOT_DB_URL is not set or the database is unreachable",
)


PROFILE = {
    "height_cm": 180,
    "weight_kg": 78,
    "age_years": 27,
    "gender": "Male",
    "sport": "Basketball",
    "duration_min": 60,
    "activity_level": "Moderately Active",
    "goal": "Muscle Gain",
}

TARGETS = {
    "daily_calories": 3000,
    "protein_g": 177.0,
    "carbs_g": 387.0,
    "fat_g": 82.7,
    "sugar_g": 80.2,
}

MEALS = [
    {"food_name": "Test Idli", "protein": 4.5, "fat": 1.2, "carbs": 18.0},
    {"food_name": "Test Dal", "protein": 9.0, "fat": 3.4, "carbs": 22.5},
]


def _delete_plan(plan_id: int) -> None:
    with storage._connect() as connection:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM profiles WHERE profile_id = "
                    "(SELECT profile_id FROM plans WHERE plan_id = %s)",
                    (plan_id,),
                )


@pytest.fixture
def saved_plan():
    plan_id = storage.save_plan(PROFILE, TARGETS, MEALS, predicted_intake_kcal=2540)
    yield plan_id
    _delete_plan(plan_id)


def test_save_plan_returns_an_id(saved_plan):
    assert isinstance(saved_plan, int)
    assert saved_plan > 0


def test_saved_plan_keeps_its_foods_in_order(saved_plan):
    items = storage.plan_items(saved_plan)
    assert [item["food_name"] for item in items] == ["Test Idli", "Test Dal"]
    assert float(items[0]["protein_g"]) == pytest.approx(4.5)


def test_saved_plan_appears_in_recent_plans(saved_plan):
    rows = storage.recent_plans(limit=20)
    match = next((row for row in rows if row["plan_id"] == saved_plan), None)

    assert match is not None, "the saved plan should appear in the history"
    assert match["food_count"] == len(MEALS)
    assert match["sport"] == PROFILE["sport"]
    assert match["daily_calories"] == TARGETS["daily_calories"]


def test_deleting_the_profile_cascades_to_plan_and_items(saved_plan):
    """A profile owns its plans, and a plan owns its foods.

    Deleting the profile must leave nothing orphaned behind it.
    """
    assert storage.plan_items(saved_plan), "fixture should have created foods"

    _delete_plan(saved_plan)

    assert storage.plan_items(saved_plan) == []
    assert all(row["plan_id"] != saved_plan for row in storage.recent_plans(limit=50))


def test_invalid_profile_is_rejected_and_saves_nothing():
    """A CHECK violation must roll the whole save back.

    Age 200 fails the profiles CHECK, so no profile, plan or food rows may
    survive the attempt.
    """
    import psycopg2

    before = {row["plan_id"] for row in storage.recent_plans(limit=100)}

    bad_profile = dict(PROFILE, age_years=200)
    with pytest.raises(psycopg2.Error):
        storage.save_plan(bad_profile, TARGETS, MEALS)

    after = {row["plan_id"] for row in storage.recent_plans(limit=100)}
    assert before == after, "a rejected save must not leave rows behind"
