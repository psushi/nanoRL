import os

import pytest

os.environ.setdefault("MUJOCO_GL", "disable")


def pytest_addoption(parser):
    parser.addoption("--full", action="store_true", help="also run the slow humanoid and terrain tasks")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--full"):
        return
    skip = pytest.mark.skip(reason="slow on CPU; run with --full")
    for item in items:
        if "full" in item.keywords:
            item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "full: slow test, runs only with --full")
