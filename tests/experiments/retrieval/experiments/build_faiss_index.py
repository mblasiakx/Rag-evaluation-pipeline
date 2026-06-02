import sys
import json
import faiss
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

EMBEDDINGS_BASE_DIR = Path("tests/experiments/embedding/experiments/embedding_quality")
RETRIEVAL_BASE_DIR = Path("tests/experiments/retrieval/retrieval_experiments_results")


def select_faiss_index(experiment_name: str, dim: int):
    if "no_normalization" in experiment_name.lower():
        return faiss.IndexFlatL2(dim)
    return faiss.IndexFlatIP(dim)


def main():
    RETRIEVAL_BASE_DIR.mkdir(exist_ok=True)

    for embedding_experiment_dir in EMBEDDINGS_BASE_DIR.iterdir():
        if not embedding_experiment_dir.is_dir() or embedding_experiment_dir.name.startswith("_"):
            continue

        experiment_name = embedding_experiment_dir.name
        print(f"\n[EXPERIMENT] {experiment_name}")

        retrieval_experiment_dir = RETRIEVAL_BASE_DIR / experiment_name
        retrieval_experiment_dir.mkdir(exist_ok=True)

        for chunk_config_dir in embedding_experiment_dir.iterdir():
            if not chunk_config_dir.is_dir():
                continue

            chunk_config_name = chunk_config_dir.name
            print(f"  [CHUNKS] {chunk_config_name}")

            embeddings_path = chunk_config_dir / "embeddings.npy"
            metadata_path = chunk_config_dir / "metadata.json"

            if not embeddings_path.exists() or not metadata_path.exists():
                print("    [SKIP] Brak embeddings.npy lub metadata.json")
                continue

            embeddings = np.load(embeddings_path).astype("float32")

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            chunk_ids = [c["chunk_id"] for c in metadata["chunks"]]

            if len(chunk_ids) != embeddings.shape[0]:
                raise ValueError(
                    f"Mismatch embeddings ({embeddings.shape[0]}) "
                    f"vs chunk_ids ({len(chunk_ids)})"
                )

            dim = embeddings.shape[1]
            index = select_faiss_index(experiment_name, dim)
            index.add(embeddings)

            out_dir = retrieval_experiment_dir / chunk_config_name
            out_dir.mkdir(parents=True, exist_ok=True)

            faiss.write_index(index, str(out_dir / "faiss.index"))

            index_mapping = {
                "faiss_id_to_chunk_id": {
                    str(i): chunk_id for i, chunk_id in enumerate(chunk_ids)
                }
            }

            with open(out_dir / "index_mapping.json", "w", encoding="utf-8") as f:
                json.dump(index_mapping, f, indent=2)

            print(f"    [OK] FAISS index zapisany do {out_dir}")


if __name__ == "__main__":
    main()
