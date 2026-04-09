from app import app


def test_health_endpoint():
    """
    Simple health check test.

    This verifies that:
    - the /health endpoint exists
    - it returns HTTP 200
    - it returns the expected JSON payload
    """
    client = app.test_client()

    response = client.get("/health")
    data = response.get_json()

    assert response.status_code == 200
    assert data is not None
    assert data["status"] == "ok"
    assert data["service"] == "rag-policy-assistant"