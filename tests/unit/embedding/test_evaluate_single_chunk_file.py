import pytest
from pathlib import Path                                                                                                                                                                                     
import yaml 
import json
#from tests.experiments.embedding.experiments.embedding_quality.evaluate_single_chunk_file import evaluate_single_chunk_file


def load_chunks(chunk_file):
      texts = []
      chunks_meta = []
      with open(chunk_file, "r", encoding="utf-8") as f:
          for line in f:
              row = json.loads(line)
              texts.append(row["text"])
              chunks_meta.append({
                  "chunk_id": row["chunk_id"],
                  "doc_id": row["doc_id"],
                  "start_char": row["start_char"],
                  "end_char": row["end_char"]
              })
      return texts, chunks_meta


def test_loads_texts_from_valid_jsonl(tmp_path):
    jsonl = tmp_path / "chunks.jsonl"
    jsonl.write_text(
          '{"chunk_id": "c0", "doc_id": "doc1", "text": "hello", "start_char": 0, "end_char": 5}\n')
    texts, chunks_meta = load_chunks(jsonl)
    assert texts[0] == "hello"
    assert chunks_meta[0]["chunk_id"] == "c0"
    assert chunks_meta[0]["doc_id"] == "doc1"
    assert chunks_meta[0]["start_char"] == 0                                                                                                                                                                     
    assert chunks_meta[0]["end_char"] == 5  
    assert len(texts) == len(chunks_meta) == 1 
 


