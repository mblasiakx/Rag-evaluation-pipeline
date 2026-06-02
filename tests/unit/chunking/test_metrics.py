from tests.experiments.chunking.metrics.boundary_quality import boundary_quality
from tests.experiments.chunking.metrics.chunk_size_stats import chunk_size_mean
from tests.experiments.chunking.metrics.chunk_size_stats import chunk_size_std

def test_boundary_quality_all_chunks_end_with_punctuation():
    chunks = [
    "Lyon is a city in France.",
    "It is located in the Auvergne region.",
    "The city has a rich history."
]
    result = boundary_quality(chunks)
    assert result == 1.0

def test_boundary_quality_no_chunks_end_with_punctuation():
    chunks = [
    "Lyon is a city in",
    "located in the Auvergne",
    "The city has a rich"
]
    result = boundary_quality(chunks)
    assert result == 0.0

def test_boundary_quality_mixed():                                                                                                                                                                           
      chunks = ["Lyon is a city.", "Lyon is a city"]
      result = boundary_quality(chunks)                                                                                                                                                                        
      assert result == 0.5 

def test_boundary_quality_empty_list():
    chunks = []
    result = boundary_quality(chunks)
    assert result == 0.0



def test_chunk_size_mean():
    chunks = ["a" * 10, "a" * 10, "a" * 10]
    result = chunk_size_mean(chunks)
    assert result == 10

def test_chunk_size_std():
    chunks = ["a" * 10, "a" * 10, "a" * 10]
    result = chunk_size_std(chunks)
    assert result == 0.0

def test_chunk_size_std_with_one_chunk():
    chunks = ["a" * 10]
    result = chunk_size_std(chunks)
    assert result == 0.0

def test_chunk_size_std_with_different_sizes():
    chunks = ["a" * 10, "a" * 20]
    result = chunk_size_std(chunks)
    assert result > 0.0

def test_chunk_size_mean_empty():
    assert chunk_size_mean([]) == 0.0

def test_chunk_size_std_empty():
    assert chunk_size_std([]) == 0.0

def test_boundary_quality_question_and_exclamation():
    chunks = ["Is Lyon in France?", "Yes it is!", "Lyon is a city."]
    result = boundary_quality(chunks)
    assert result == 1.0
