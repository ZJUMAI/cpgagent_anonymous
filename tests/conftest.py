from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("MEDCLAW_RUN_PRIVATE_DATA_TESTS") == "1":
        return
    skip = pytest.mark.skip(
        reason="requires private data; set MEDCLAW_RUN_PRIVATE_DATA_TESTS=1 to run"
    )
    for item in items:
        if "private_data" in item.keywords:
            item.add_marker(skip)
