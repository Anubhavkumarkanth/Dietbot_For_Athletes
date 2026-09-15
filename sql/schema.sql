-- ============================================================================
--  DietBot - PostgreSQL schema
--
--  Run with:  psql -U dietbot_app -d dietbot -f sql/schema.sql
--
--  Three tables. The app previously saved nothing at all, so a user could not
--  look back at a plan they had generated. This stores the profile a plan was
--  built from, the plan itself, and the foods in it.
-- ============================================================================

DROP TABLE IF EXISTS plan_items CASCADE;
DROP TABLE IF EXISTS plans      CASCADE;
DROP TABLE IF EXISTS profiles   CASCADE;


-- ----------------------------------------------------------------------------
-- profiles
-- The inputs a plan was generated from. Stored separately from the plan so the
-- same profile can be reused across several plans, and so a saved plan can be
-- explained later ("what did I enter to get this?").
-- ----------------------------------------------------------------------------
CREATE TABLE profiles (
    profile_id     SERIAL PRIMARY KEY,
    height_cm      NUMERIC(5, 1) NOT NULL CHECK (height_cm  BETWEEN 100 AND 250),
    weight_kg      NUMERIC(5, 1) NOT NULL CHECK (weight_kg  BETWEEN 30  AND 300),
    age_years      INT           NOT NULL CHECK (age_years  BETWEEN 10  AND 100),
    gender         TEXT          NOT NULL CHECK (gender IN ('Male', 'Female')),
    sport          TEXT          NOT NULL,
    duration_min   INT           NOT NULL CHECK (duration_min > 0),
    activity_level TEXT          NOT NULL,
    goal           TEXT          NOT NULL,
    created_at     TIMESTAMPTZ   NOT NULL DEFAULT now()
);


-- ----------------------------------------------------------------------------
-- plans
-- One generated plan: the calorie target and the macro targets derived from it.
--
-- predicted_intake_kcal is the RandomForest's estimate of typical intake for
-- this profile. It is stored alongside the target, not instead of it, because
-- they answer different questions - see README, "What the model predicts".
-- It is nullable because the model is optional: the app still works if the
-- model artifact is missing.
-- ----------------------------------------------------------------------------
CREATE TABLE plans (
    plan_id               SERIAL PRIMARY KEY,
    profile_id            INT NOT NULL REFERENCES profiles (profile_id) ON DELETE CASCADE,
    daily_calories        INT NOT NULL CHECK (daily_calories > 0),
    predicted_intake_kcal INT CHECK (predicted_intake_kcal IS NULL OR predicted_intake_kcal > 0),
    protein_g             NUMERIC(6, 1) NOT NULL CHECK (protein_g >= 0),
    carbs_g               NUMERIC(6, 1) NOT NULL CHECK (carbs_g   >= 0),
    fat_g                 NUMERIC(6, 1) NOT NULL CHECK (fat_g     >= 0),
    sugar_g               NUMERIC(6, 1) NOT NULL CHECK (sugar_g   >= 0),
    created_at            TIMESTAMPTZ   NOT NULL DEFAULT now()
);


-- ----------------------------------------------------------------------------
-- plan_items
-- The foods chosen for a plan.
--
-- ON DELETE CASCADE: a food line has no meaning without its plan.
--
-- The macro columns copy the food's values at the time the plan was built,
-- rather than referencing the food table. The food data is a static CSV today,
-- but if it is ever updated, a saved plan must still show the numbers it was
-- actually built from.
--
-- position preserves the order the foods were chosen in, which is lost if rows
-- are read back without an ORDER BY.
-- ----------------------------------------------------------------------------
CREATE TABLE plan_items (
    plan_item_id SERIAL PRIMARY KEY,
    plan_id      INT  NOT NULL REFERENCES plans (plan_id) ON DELETE CASCADE,
    food_name    TEXT NOT NULL,
    protein_g    NUMERIC(6, 2) NOT NULL CHECK (protein_g >= 0),
    fat_g        NUMERIC(6, 2) NOT NULL CHECK (fat_g     >= 0),
    carbs_g      NUMERIC(6, 2) NOT NULL CHECK (carbs_g   >= 0),
    position     INT  NOT NULL CHECK (position >= 0),
    UNIQUE (plan_id, position)
);


-- Plans are always read back newest-first for a given profile, and the history
-- view lists the most recent plans overall.
CREATE INDEX idx_plans_created_at ON plans (created_at DESC);
CREATE INDEX idx_plans_profile    ON plans (profile_id);
