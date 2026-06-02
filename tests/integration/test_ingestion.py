import pytest
import math


# --- load_document ---

def test_document_is_string(document):
    assert isinstance(document, str)


def test_document_is_not_empty(document):
    assert len(document.strip()) > 0


# --- load_document + chunker ---

def test_chunks_are_produced(chunks):
    assert len(chunks) > 0


def test_chunks_have_required_keys(chunks):
    for chunk in chunks:
        assert "text" in chunk
        assert "start_char" in chunk
        assert "end_char" in chunk


def test_chunks_are_non_empty(chunks):
    assert all(len(c["text"]) > 0 for c in chunks)


# --- config params flow into chunker ---

def test_config_chunk_size_applied(config, chunks):
    c = config["chunking"]
    max_size = c.get("chunk_size") or c.get("max_chunk_size")
    assert all(len(chunk["text"]) <= max_size for chunk in chunks)


def test_config_overlap_applied(config, chunks):
    if config["chunking"]["type"] != "fixed":
        pytest.skip("overlap test only applies to fixed chunker")
    chunk_size = config["chunking"]["chunk_size"]
    overlap = config["chunking"]["overlap"]
    assert chunks[1]["start_char"] == chunk_size - overlap


def test_chunk_count_matches_config(config, document, chunks):
    if config["chunking"]["type"] != "fixed":
        pytest.skip("chunk count formula only applies to fixed chunker")
    chunk_size = config["chunking"]["chunk_size"]
    overlap = config["chunking"]["overlap"]
    step = chunk_size - overlap
    expected = math.ceil(len(document) / step)
    assert len(chunks) == expected


def test_chunk_text_matches_document_offsets(document, chunks):
    for chunk in chunks:
        assert document[chunk["start_char"]:chunk["end_char"]] == chunk["text"]


def test_chunk_continuity_fixed(config, chunks):
    if config["chunking"]["type"] != "fixed":
        pytest.skip("only for fixed chunker")
    step = config["chunking"]["chunk_size"] - config["chunking"]["overlap"]
    for i in range(1, len(chunks)):
        assert chunks[i]["start_char"] == chunks[i - 1]["start_char"] + step
