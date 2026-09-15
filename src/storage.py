"""PostgreSQL persistence for generated plans.

Written with psycopg2 and hand-written SQL rather than an ORM. For three tables
and four queries an ORM would be more machinery than the problem needs, and
keeping the SQL visible makes what the app actually sends to the database
obvious.

Saving a plan writes to three tables and is done in a single transaction: a
profile with no plan, or a plan with no foods, would be a half-saved record that
the history view could not display. psycopg2 opens a transaction implicitly, so
the `with connection` block below commits on success and rolls back if any
statement raises.

Persistence is optional. If DIETBOT_DB_URL is not set the app runs exactly as it
did before, just without a history - `is_available` is what the UI checks.
"""

import os
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor

ENV_VAR = "DIETBOT_DB_URL"


def database_url() -> str | None:
    url = os.getenv(ENV_VAR)
    return url if url else None


def is_available() -> bool:
    """True when a database is configured and actually reachable.

    Connecting is the only honest test: a URL can be set and still point at a
    server that is not running, and the UI needs to know which it is before
    offering to save anything.
    """
    url = database_url()
    if not url:
        return False
    try:
        with psycopg2.connect(url, connect_timeout=5) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
        return True
    except psycopg2.Error:
        return False


@contextmanager
def _connect():
    url = database_url()
    if not url:
        raise RuntimeError(f"{ENV_VAR} is not set")
    connection = psycopg2.connect(url, connect_timeout=5)
    try:
        yield connection
    finally:
        connection.close()


def save_plan(profile: dict, targets: dict, meals: list[dict],
              predicted_intake_kcal: float | None = None) -> int:
    """Save a profile, the plan generated from it, and its foods.

    All three inserts share one transaction, so a plan is either stored
    completely or not at all.

    Args:
        profile: height_cm, weight_kg, age_years, gender, sport, duration_min,
            activity_level, goal.
        targets: daily_calories plus protein_g / carbs_g / fat_g / sugar_g.
        meals: the chosen foods, each with food_name, protein, fat, carbs.
        predicted_intake_kcal: the model's estimate, or None if unavailable.

    Returns:
        The generated plan_id.
    """
    with _connect() as connection:
        with connection:  # commits on exit, rolls back on exception
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO profiles (height_cm, weight_kg, age_years, gender,
                                          sport, duration_min, activity_level, goal)
                    VALUES (%(height_cm)s, %(weight_kg)s, %(age_years)s, %(gender)s,
                            %(sport)s, %(duration_min)s, %(activity_level)s, %(goal)s)
                    RETURNING profile_id
                    """,
                    profile,
                )
                profile_id = cursor.fetchone()[0]

                cursor.execute(
                    """
                    INSERT INTO plans (profile_id, daily_calories, predicted_intake_kcal,
                                       protein_g, carbs_g, fat_g, sugar_g)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING plan_id
                    """,
                    (
                        profile_id,
                        int(targets["daily_calories"]),
                        int(predicted_intake_kcal) if predicted_intake_kcal else None,
                        targets["protein_g"],
                        targets["carbs_g"],
                        targets["fat_g"],
                        targets["sugar_g"],
                    ),
                )
                plan_id = cursor.fetchone()[0]

                cursor.executemany(
                    """
                    INSERT INTO plan_items (plan_id, food_name, protein_g, fat_g, carbs_g, position)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (plan_id, meal["food_name"], meal["protein"], meal["fat"],
                         meal["carbs"], position)
                        for position, meal in enumerate(meals)
                    ],
                )

    return plan_id


def recent_plans(limit: int = 5) -> list[dict]:
    """The most recent saved plans, with the profile each was built from.

    A LEFT JOIN onto plan_items so a plan still appears if it somehow has no
    foods, and COUNT over the joined rows gives the number of foods per plan
    without a second query per row.
    """
    with _connect() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT p.plan_id,
                       p.created_at,
                       p.daily_calories,
                       p.predicted_intake_kcal,
                       p.protein_g,
                       p.carbs_g,
                       p.fat_g,
                       pr.sport,
                       pr.goal,
                       pr.weight_kg,
                       COUNT(pi.plan_item_id) AS food_count
                FROM plans p
                JOIN profiles   pr ON pr.profile_id = p.profile_id
                LEFT JOIN plan_items pi ON pi.plan_id = p.plan_id
                GROUP BY p.plan_id, pr.profile_id
                ORDER BY p.created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]


def plan_items(plan_id: int) -> list[dict]:
    """The foods in one saved plan, in the order they were chosen."""
    with _connect() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT food_name, protein_g, fat_g, carbs_g
                FROM plan_items
                WHERE plan_id = %s
                ORDER BY position
                """,
                (plan_id,),
            )
            return [dict(row) for row in cursor.fetchall()]
