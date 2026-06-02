import pytest
pytestmark = pytest.mark.slow
from tests.experiments.chunking.metrics.inter_chunk_similarity import inter_chunk_similarity
from tests.experiments.chunking.metrics.topic_drift import topic_drift
from tests.experiments.chunking.metrics.chunk_entropy import chunk_entropy
from tests.experiments.chunking.metrics.semantic_coherence import semantic_coherence


def test_inter_chunk_similarity_high():
    chunks = ["Lyon is a city in France.", "Lyon is located in the Auvergne region. Lyon has a large population."]
    result = inter_chunk_similarity(chunks)
    assert result > 0.6

def test_inter_chunk_similarity_low():
    chunks = ["Lyon is a city in France.", "The moon orbits the Earth. Python is a programming language."]
    result = inter_chunk_similarity(chunks)
    assert result < 0.5

def test_topic_drift_high():
    chunks = ["Lyon is a city in France.", "The moon orbits the Earth. Python is a programming language."]
    result = topic_drift(chunks, query="Sports and health")
    assert result > 0.8

def test_topic_drift_low():
    chunks = ["Lyon is a city in France.", "Lyon has a rich history and culture."]
    result = topic_drift(chunks, query="Information about Lyon")
    assert result < 0.5

def test_topic_drift_empty_chunks():
    result = topic_drift([], query="Information about Lyon")
    assert result == 0.0

def test_topic_drift_empty_query():
    with pytest.raises(ValueError):
        topic_drift(["Lyon is a city in France."], query="")

def test_chunk_entropy_high():
    # każdy chunk musi mieć ≥2 zdania rozdzielone ". " żeby metryka mogła policzyć entropię
    chunks = [
        "Lyon is in France. I like apples. Cars are fast.",
        "The moon is bright. Python is a language. Paris is beautiful.",
    ]
    result = chunk_entropy(chunks)
    assert result > 0.7

def test_chunk_entropy_low():
    chunks = ["Lyon is a city in France.", "Lyon i a big city and It's interesting for turists."]
    result = chunk_entropy(chunks)
    assert result < 0.5

def test_semantic_coherence_high():
    chunks = ["Lyon is a city in France. Lyon is a big city. Lyon has a rich history."]
    result = semantic_coherence(chunks)
    assert result > 0.7

def test_semantic_coherence_low():
    chunks = ["Lyon is a city in France. The moon orbits the Earth. Python is a programming language."]
    result = semantic_coherence(chunks)
    assert result < 0.5

def test_semantic_coherence_empty_list():
    chunks = []
    result = semantic_coherence(chunks)
    assert result == 0.0

def test_semantic_coherence_single_sentence():
    chunks = ["Lyon is a city in France"]
    result = semantic_coherence(chunks)
    assert result == 1.0

def test_semantic_coherence_averages_across_chunks():
    coherent = "Lyon is a city in France. Lyon has a big population."
    incoherent = "Lyon is a city. The moon orbits Earth. Python is fast."
    result_coherent = semantic_coherence([coherent])
    result_incoherent = semantic_coherence([incoherent])
    result_mixed = semantic_coherence([coherent, incoherent])
    assert result_incoherent < result_mixed < result_coherent

def test_semantic_coherence_returns_value_in_range():
    chunks = ["Lyon is a city. The moon is bright. Python is fast."]
    result = semantic_coherence(chunks)
    assert 0.0 <= result <= 1.0

def test_inter_chunk_similarity_single_chunk():
    result = inter_chunk_similarity(["Lyon is a city in France."])
    assert result == 0.0

def test_chunk_entropy_empty_list():
    result = chunk_entropy([])
    assert result == 0.0
