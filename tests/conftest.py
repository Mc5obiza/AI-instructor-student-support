from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import helper, main


@pytest.fixture
def api_client() -> TestClient:
    main.app.dependency_overrides.clear()
    with TestClient(main.app) as client:
        yield client
    main.app.dependency_overrides.clear()


@pytest.fixture
def authenticated_client() -> TestClient:
    main.app.dependency_overrides.clear()
    main.app.dependency_overrides[main.get_current_user] = lambda: helper.CurrentUser(
        user_id="test-user",
        email="student@example.test",
    )
    with TestClient(main.app) as client:
        yield client
    main.app.dependency_overrides.clear()
