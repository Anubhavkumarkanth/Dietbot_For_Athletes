-- ============================================================================
--  DietBot - DELETE ALL SAVED DATA and recreate the schema
--
--  Drops every table, so every saved profile, plan and food line is lost.
--  Run it from the repository root, only when that is what you want:
--
--      psql -U dietbot_app -d dietbot -f sql/reset.sql
--
--  For a first-time setup, or to make sure the tables exist, use schema.sql.
-- ============================================================================

DROP TABLE IF EXISTS plan_items CASCADE;
DROP TABLE IF EXISTS plans      CASCADE;
DROP TABLE IF EXISTS profiles   CASCADE;

\i sql/schema.sql
