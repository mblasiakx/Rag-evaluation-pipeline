"""Ten plik wykonuje faktyczne: wczytanie chunków, embedding, metryki, zapis wyników"""

import sys
from pathlib import Path
import json
import numpy as np

# Dodaj root projektu (katalog z folderem rag/) do sys.path
_ROOT = Path(__file__).resolve()
while not (_ROOT / "rag").exists():
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))

from rag.embedder import build_model
from tests.experiments.embedding.metrics.knn_neighbors import compute_knn_neighbors
from tests.experiments.embedding.metrics.intra_similarity import compute_intra_similarity
from tests.experiments.embedding.metrics.anisotropy import compute_anisotropy
from tests.experiments.embedding.metrics.embedding_variance import compute_embedding_variance
from tests.experiments.embedding.metrics.recall_at_k import compute_recall_at_k
from tests.experiments.embedding.metrics.sequential_similarity import compute_sequential_similarity


def evaluate_single_chunk_file(chunk_file, out_dir, config):
    chunks_meta = []
    texts = []

    with open(chunk_file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            chunks_meta.append({
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "start_char": row["start_char"],
                "end_char": row["end_char"],
            })

    model_cfg = config["model"]
    model = build_model(
        model_name=model_cfg["name"],
        pooling=model_cfg["pooling"],
    )

    embeddings = model.encode(
        texts,
        batch_size=model_cfg["batch_size"],
        convert_to_numpy=True,
        normalize_embeddings=model_cfg["normalize"],
    )

    if config["outputs"]["save_embeddings"]:
        np.save(out_dir / "embeddings.npy", embeddings)
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({"chunks": chunks_meta}, f, ensure_ascii=False, indent=2)

    chunk_ids = [c["chunk_id"] for c in chunks_meta]
    doc_ids = [c["doc_id"] for c in chunks_meta]
    results = {}

    if config["metrics"]["compute_intra_similarity"]:
        results["intra_similarity"] = compute_intra_similarity(embeddings, doc_ids)

    if config["metrics"]["compute_knn_neighbors"]:
        k = config["metrics"]["k_neighbors"]
        results["knn"] = compute_knn_neighbors(embeddings, chunk_ids, k=k)

    if config["metrics"].get("compute_anisotropy"):
        results["anisotropy"] = compute_anisotropy(embeddings)

    if config["metrics"].get("compute_embedding_variance"):
        results["embedding_variance"] = compute_embedding_variance(embeddings)

    if config["metrics"].get("compute_recall_at_k"):
        k = config["metrics"]["k_neighbors"]
        results["recall_at_k"] = compute_recall_at_k(embeddings, doc_ids, k=k)

    if config["metrics"].get("compute_sequential_similarity"):
        results["sequential_similarity"] = compute_sequential_similarity(embeddings)

    with open(out_dir / "results.json", "w") as f:
        json.dump(
            results,
            f,
            indent=2,
            default=lambda o: o.tolist() if isinstance(o, np.ndarray)
                              else float(o) if isinstance(o, (np.floating, np.integer))
                              else str(o),
        )

    print(f"[OK] Saved results to {out_dir}")
