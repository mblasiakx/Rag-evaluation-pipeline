import pytest
import os
import sys
from contextlib import asynccontextmanager
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import api.main as api_main


@pytest.fixture(scope="module")
def client():
    mock_doc = MagicMock()
    mock_doc.page_content = "The European Cup was the tournament before Champions League."

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [mock_doc, mock_doc]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "The European Cup was before Champions League."

    @asynccontextmanager
    async def mock_lifespan(app):
        api_main.pipeline["retriever"] = mock_retriever
        api_main.pipeline["llm"] = mock_llm
        api_main.pipeline["prompt_fn"] = lambda q, ctx: f"{q}\n{ctx}"
        yield
        api_main.pipeline.clear()

    api_main.app.router.lifespan_context = mock_lifespan

    with TestClient(api_main.app) as c:
        yield c


def test_ask_returns_200(client):
    response = client.post("/ask", json={"question": "Who won Champions League in 2012?"})
    assert response.status_code == 200


def test_ask_response_has_required_fields(client):
    response = client.post("/ask", json={"question": "Who won Champions League in 2012?"})
    data = response.json()
    assert "answer" in data
    assert "contexts" in data
    assert "latency_ms" in data


def test_ask_answer_is_non_empty_string(client):
    response = client.post("/ask", json={"question": "Who won Champions League in 2012?"})
    data = response.json()
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0


def test_ask_contexts_is_list_of_strings(client):
    response = client.post("/ask", json={"question": "Who won Champions League in 2012?"})
    data = response.json()
    assert isinstance(data["contexts"], list)
    assert len(data["contexts"]) > 0
    assert all(isinstance(c, str) for c in data["contexts"])


def test_ask_empty_question_returns_422(client):
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422
