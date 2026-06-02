import pytest


def test_qa_data_match_format(qa_data):
    assert isinstance(qa_data, list)
    assert len(qa_data) > 0
    for item in qa_data:
        assert "question" in item
        assert "answer" in item
        assert "retrieved_contexts" in item
        assert "formatted_contexts" in item
        assert "reference" in item
        assert "latency_ms" in item


def test_retrieved_contexts_are_non_empty(qa_data):
    for item in qa_data:
        assert len(item["retrieved_contexts"]) > 0
        assert all(isinstance(c, str) for c in item["retrieved_contexts"])


def test_latency_is_recorded(qa_data):
    for item in qa_data:
        assert isinstance(item["latency_ms"], (int, float))
        assert item["latency_ms"] >= 0
