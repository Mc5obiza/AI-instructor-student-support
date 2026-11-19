from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import main

pytestmark = pytest.mark.unit


def test_app_startup_initializes_retrieval_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    initialized: list[bool] = []
    monkeypatch.setattr(
        main,
        "initialize_retrieval_runtime",
        lambda: initialized.append(True),
    )

    with TestClient(main.create_app()):
        assert initialized == [True]
