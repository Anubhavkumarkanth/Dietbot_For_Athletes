"""Pytest configuration.

The persistence tests write and delete rows, so they only run against a database
whose name ends in _test, given as DIETBOT_TEST_DB_URL. The app's own
DIETBOT_DB_URL is removed from the environment for the whole run, so a test can
never reach the database you actually use. Without DIETBOT_TEST_DB_URL the
persistence tests are skipped.

(This file used to load .env so the tests could find DIETBOT_DB_URL. That ran
them against the real database, which is the opposite of what a test should do.)
"""

import os
from urllib.parse import urlparse

import pytest


def pytest_configure(config):
    os.environ.pop("DIETBOT_DB_URL", None)

    test_url = os.environ.get("DIETBOT_TEST_DB_URL")
    if not test_url:
        return

    name = urlparse(test_url).path.lstrip("/")
    if not name.endswith("_test"):
        pytest.exit(
            f"Refusing to run: DIETBOT_TEST_DB_URL points at '{name}', and the persistence "
            "tests write and delete rows. Use a database whose name ends in _test.",
            returncode=2,
        )
    os.environ["DIETBOT_DB_URL"] = test_url
