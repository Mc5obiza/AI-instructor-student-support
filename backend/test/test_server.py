from fastapi.testclient import TestClient
import pytest

from backend.main import app

pytestmark = pytest.mark.feature

client = TestClient(app)

def test_app_starts():
    response = client.get("/docs")
    assert response.status_code == 200