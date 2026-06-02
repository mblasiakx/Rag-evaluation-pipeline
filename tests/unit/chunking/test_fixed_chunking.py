import pytest
from tests.experiments.chunking.chunkers.fixed_chunking import FixedChunking

@pytest.fixture
def chunker():
    return FixedChunking(chunk_size=10, overlap=0)

def test_correct_number_of_chunks(chunker):
    text = "a" * 30  
    result = chunker.chunk(text)
    assert len(result) == 3

def test_correct_overlap():
    text = "a" * 30
    overlap_chunker = FixedChunking(chunk_size=10, overlap=5)
    result = overlap_chunker.chunk(text)
    # step=5, 30 chars → starts at 0,5,10,15,20,25 → 6 chunks (last partial)
    assert len(result) == 6


def test_chunk_offsets_no_overlap():
    text = "abcdefghij"
    chunker = FixedChunking(chunk_size=5, overlap=0) 
    result = chunker.chunk(text)
    assert result[0]["start_char"] ==0
    assert result[0]["end_char"] ==5 
    assert result[1]["start_char"] ==5
    assert result[1]["end_char"] ==10 
    
def test_chunk_offsets_overlap():
    text = "abcdefghijklmnopqrst"
    chunker = FixedChunking(chunk_size=10, overlap=5) 
    result = chunker.chunk(text)
    assert result[0]["start_char"] ==0
    assert result[0]["end_char"] ==10 
    assert result[1]["start_char"] ==5
    assert result[1]["end_char"] ==15 

def test_chunk_equal_to_text_size():
    text = "a" * 10
    chunker = FixedChunking(chunk_size=10, overlap=0) 
    result = chunker.chunk(text)
    assert len(result)==1
    assert result[0]["start_char"] ==0
    assert result[0]["end_char"] ==10 


def test_max_valid_overlap_does_not_raise():
    text = "a" * 12
    chunker = FixedChunking(chunk_size=10, overlap=9) 
    result = chunker.chunk(text)
    assert len(result)==12
  
#edge cases ----------
def test_empty_document(chunker):
    result = chunker.chunk("")
    assert result == []


def test_last_chunk_shorter_than_chunk_size(chunker):
    text = "a" * 25  
    result = chunker.chunk(text)
    assert len(result[-1]["text"]) < 10
   
def test_text_shorter_than_chunk_size(chunker):
    text = "a" * 5  
    result = chunker.chunk(text)
    assert len(result) == 1
    assert len(result[0]["text"]) == 5

def test_overlap_bigger_than_chunk_size():
    with pytest.raises(ValueError):
        FixedChunking(chunk_size=10, overlap=30)

def test_overlap_equal_to_chunk_size_raises():
    with pytest.raises(ValueError):
        FixedChunking(chunk_size=10, overlap=10)

def test_non_string_input_is_converted():
    chunker = FixedChunking(chunk_size=5, overlap=0)
    result = chunker.chunk(12345)
    assert isinstance(result, list)
    assert result[0]["text"] == "12345"
