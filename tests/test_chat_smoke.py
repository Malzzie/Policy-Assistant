from app import app


def test_chat_endpoint_requires_question():
    """
    Verify that /chat returns 400 if no question is provided.
    """
    client = app.test_client()

    response = client.post("/chat", json={})
    data = response.get_json()

    assert response.status_code == 400
    assert data is not None
    assert "error" in data


def test_chat_endpoint_smoke():
    """
    Smoke test for the /chat endpoint.

    We do not enforce exact answer text here because:
    - the system may run in llm mode
    - or fallback mode
    - or retrieval results may evolve

    We only verify the endpoint works and returns the expected structure.
    """
    client = app.test_client()

    response = client.post(
        "/chat",
        json={"question": "How many PTO days can employees carry over?"},
    )

    data = response.get_json()

    assert response.status_code == 200
    assert data is not None

    assert "question" in data
    assert "answer" in data
    assert "citations" in data
    assert "mode" in data

    assert data["question"] == "How many PTO days can employees carry over?"
    assert isinstance(data["answer"], str)
    assert isinstance(data["citations"], list)
    assert isinstance(data["mode"], str)