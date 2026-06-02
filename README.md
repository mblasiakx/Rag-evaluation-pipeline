# RAG Evaluation Pipeline

A research project studying how different pipeline configurations affect answer quality in Retrieval-Augmented Generation (RAG) systems. The pipeline loads documents, chunks them, builds a FAISS vector index, generates answers via a local LLM, and evaluates quality using RAGAS metrics and a custom completeness score. Three configurations are systematically compared.

---

## Project Goal

Most RAG tutorials show how to build a pipeline — this project focuses on **how to evaluate and compare them**. Each component (chunking strategy, embedding model, retrieval depth) is treated as an experimental variable with measurable impact on answer quality.

---

## Pipeline Architecture

```
Document (txt/pdf/docx)
  → DocumentLoader          rag/file_loader.py
  → Chunker                 rag/chunking/  (FixedChunker or SemanticChunker)
  → HuggingFace Embeddings  rag/embedder.py  (all-MiniLM-L6-v2, explicit pooling)
  → FAISS Vector Store      rag/retriever.py
  → Ollama LLM              rag/generator.py  (gemma3:1b)
  → Evaluator               evaluation/evaluator.py  (RAGAS + completeness)
  → MLflow + CSV            experiment tracking
```

---

## Experiments

### Chunking

Two strategies were compared:

| Strategy | Description |
|---|---|
| **FixedChunker** | Splits text into fixed-size character windows with configurable overlap. Predictable, fast. |
| **SemanticChunker** | Groups sentences by cosine similarity. Chunks follow topic boundaries rather than character counts. |

Key insight: larger chunks (512 chars) provide more context per retrieved document at the cost of retrieval precision.

### Embedding

Model: `sentence-transformers/all-MiniLM-L6-v2` across all configs.

Variables tested:
- **Pooling mode** — `mean` pooling applied explicitly via `SentenceTransformerEmbeddings` wrapper
- **Normalization** — `normalize=True` vs `normalize=False` affects cosine similarity sharpness in FAISS

### Retrieval

FAISS flat index with configurable `k` (number of retrieved chunks). All configs use `k=2`. Retrieved chunks are formatted with numbered labels before being passed to the LLM prompt.

---

## Configuration Comparison

Three configs tested against 11 questions on the same document (Champions League history):

| Config | Chunking | Chunk size | Overlap | Normalize | Prompt |
|---|---|---|---|---|---|
| **config_A** | Fixed | 512 | 20 | False | zero_shot |
| **config_B** | Semantic | max 256 | — | True | zero_shot |
| **config_C** | Fixed | 256 | 96 | True | zero_shot |

### Results

| Config | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Completeness | Latency (ms) |
|---|---|---|---|---|---|---|
| **config_A** | **0.773** | **0.783** | **0.591** | 0.545 | **0.808** | 2466 |
| config_B | 0.682 | 0.766 | 0.545 | 0.545 | 0.747 | **715** |
| config_C | 0.636 | 0.578 | 0.500 | 0.545 | 0.690 | 898 |

**config_A wins on all quality metrics** but is 3x slower than config_B. config_B (semantic chunking) offers the best quality/latency trade-off.

Reproduce with:
```bash
python scripts/compare_configs.py
```

---

## Evaluation Metrics

| Metric | What it measures | Source |
|---|---|---|
| **Faithfulness** | Does the answer stick to the retrieved context? (hallucination detection) | RAGAS |
| **Answer Relevancy** | Does the answer actually address the question? | RAGAS |
| **Context Precision** | Are the retrieved chunks relevant to the question? | RAGAS |
| **Context Recall** | Does the retrieved context cover the ground truth answer? | RAGAS |
| **Completeness** | Cosine similarity between answer and reference answer embeddings | Custom |

RAGAS metrics use GPT-3.5-turbo as judge. Completeness uses the same embedding model as the pipeline.

---

## Tests

```
tests/
  integration/
    conftest.py           shared fixtures (config, document, chunks, retriever, embeddings, qa_data)
    test_ingestion.py     document loading + chunking (10 tests) — offset integrity, continuity, config flow
    test_retrieval.py     FAISS retriever (2 tests) — k results, non-empty strings
    test_generation.py    run_qa with mock LLM (3 tests) — output format, contexts, latency
    test_evaluation.py    evaluator with mock RAGAS (3 tests) — columns, completeness range, count
  api/
    test_ask.py           FastAPI endpoint (5 tests) — status, fields, types, empty input validation
```

Run integration tests:
```bash
pytest tests/integration/ -v
```

Run API tests:
```bash
pytest tests/api/test_ask.py -v
```

---

## API

A FastAPI endpoint wraps the best config (config_A) as a production-style service:

```bash
uvicorn api.main:app --reload
```

```
POST /ask
{"question": "Who founded the Champions League?"}

→ {"answer": "...", "contexts": ["..."], "latency_ms": 1823}
```

Interactive docs at `http://localhost:8000/docs`.

---

## Setup

**Requirements:**
- Python 3.10+
- [Ollama](https://ollama.com) running locally with `gemma3:1b` pulled
- OpenAI API key (for RAGAS evaluation)

```bash
pip install -r requirements.txt
ollama pull gemma3:1b
cp .env.example .env  # add OPENAI_API_KEY
```

**Run a pipeline config:**
```bash
python scripts/run_pipeline.py --config configs/config_A.yaml
```

**Compare all configs:**
```bash
python scripts/compare_configs.py
```

**View experiment history in MLflow:**
```bash
mlflow ui
# open http://localhost:5000
```

---

## Stack

| Component | Tool |
|---|---|
| LLM | Ollama (gemma3:1b) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | FAISS |
| Evaluation | RAGAS + scikit-learn cosine similarity |
| Experiment tracking | MLflow |
| API | FastAPI |
| Tests | pytest |
