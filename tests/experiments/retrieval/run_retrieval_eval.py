import json
import sys
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# projekt root → rag.embedder
_PROJECT_ROOT = Path(__file__).resolve()
while not (_PROJECT_ROOT / "rag").exists():
    _PROJECT_ROOT = _PROJECT_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# retrieval dir → metrics.*
sys.path.insert(0, str(Path(__file__).resolve().parent))

sys.stdout.reconfigure(encoding="utf-8")

from rag.embedder import DEFAULT_EMBEDDING_MODEL
from metrics.precision_at_k import compute_precision_at_k
from metrics.hit_rate import compute_hit_rate
from metrics.mrr import compute_mrr

INDEXES_BASE_DIR = Path("tests/experiments/retrieval/retrieval_experiments_results")
QUERIES_DIR = Path("tests/experiments/retrieval/retrieval_queries/baseline_queries.json")
EMBEDDINGS_BASE_DIR = Path("tests/experiments/embedding/experiments/embedding_quality")

TOP_K = 5


def main():
    model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    results_all = []

    for experiment_dir in INDEXES_BASE_DIR.iterdir():
        if not experiment_dir.is_dir() or experiment_dir.name.startswith("_"):
            continue

        print(f"\n[EXPERIMENT] {experiment_dir.name}")

        for chunk_dir in experiment_dir.iterdir():
            if not chunk_dir.is_dir():
                continue

            print(f"  [CHUNKS] {chunk_dir.name}")

            index_path = chunk_dir / "faiss.index"
            mapping_path = chunk_dir / "index_mapping.json"

            if not index_path.exists() or not mapping_path.exists():
                print("    [SKIP] Missing FAISS index or index_mapping.json")
                continue

            chunks_meta_path = (
                EMBEDDINGS_BASE_DIR
                / experiment_dir.name
                / chunk_dir.name
                / "metadata.json"
            )

            if not chunks_meta_path.exists():
                print(f"    [SKIP] Missing chunks metadata: {chunks_meta_path}")
                continue

            index = faiss.read_index(str(index_path))

            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)["faiss_id_to_chunk_id"]

            with open(chunks_meta_path, "r", encoding="utf-8") as f:
                chunks_meta = json.load(f)["chunks"]

            chunk_lookup = {c["chunk_id"]: c for c in chunks_meta}

            with open(QUERIES_DIR, "r", encoding="utf-8") as f:
                queries = json.load(f)

            for q in queries:
                query_text = q["query"]
                query_offsets = q["relevant_offsets"]

                normalize = "no_normalization" not in experiment_dir.name.lower()
                query_emb = model.encode([query_text], normalize_embeddings=normalize)
                distances, indices = index.search(query_emb, TOP_K)

                retrieved_chunks = []
                for faiss_id in indices[0]:
                    fid = str(faiss_id)
                    if fid not in mapping:
                        continue
                    chunk_id = mapping[fid]
                    if chunk_id not in chunk_lookup:
                        continue
                    retrieved_chunks.append(chunk_lookup[chunk_id])

                precision = compute_precision_at_k(retrieved_chunks, query_offsets, k=TOP_K)
                hit_rate = compute_hit_rate(retrieved_chunks, query_offsets)
                mrr = compute_mrr(retrieved_chunks, query_offsets)

                results_all.append({
                    "experiment":          experiment_dir.name,
                    "chunk_config":        chunk_dir.name,
                    "query_text":          query_text,
                    "precision_at_k":      precision,
                    "hit_rate":            hit_rate,
                    "mrr":                 mrr,
                    "retrieved_chunk_ids": [ch["chunk_id"] for ch in retrieved_chunks],
                })

                print(f"    Query: {query_text}")
                print(f"    Precision@{TOP_K}: {precision:.2f}  Hit Rate: {hit_rate:.2f}  MRR: {mrr:.2f}")

    output_file = INDEXES_BASE_DIR / "retrieval_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] All results saved to {output_file}")


if __name__ == "__main__":
    main()
