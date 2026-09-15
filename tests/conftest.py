"""Pytest configuration.

Loads .env before tests are collected.

Without this, src/storage.is_available() cannot see DIETBOT_DB_URL during a
test run - only app.py called load_dotenv() - so every persistence test would
silently skip even with a working database. A test that skips when it should
run is worse than one that fails, because it looks like success.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
