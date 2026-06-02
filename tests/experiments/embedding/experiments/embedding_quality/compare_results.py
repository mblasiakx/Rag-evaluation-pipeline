"""
Aggregates all results.json from embedding quality experiments
into a single comparison table. Run from any directory:

    python compare_results.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RESULTS_ROOT = Path(__file__).resolve().parent


def _mean_knn_dist(knn_data) -> float | None:
    """Mean nearest-neighbor cosine distance, skipping self (distance=0.0)."""
    if not knn_data:
        return None
    flat = np.array(knn_data[0]).flatten()
    neighbours = flat[flat > 0.0]
    return float(np.mean(neighbours)) if len(neighbours) > 0 else None


def collect_results(root: Path) -> list[dict]:
    rows = []
    for results_file in sorted(root.glob("**/results.json")):
        experiment = results_file.parent.parent.name
        chunk_config = results_file.parent.name

        with open(results_file, encoding="utf-8") as f:
            data = json.load(f)

        rows.append({
            "experiment":           experiment,
            "chunk_config":         chunk_config,
            "intra_similarity":     data.get("intra_similarity"),
            "sequential_similarity": data.get("sequential_similarity"),
            "anisotropy":           data.get("anisotropy"),
            "embedding_variance":   data.get("embedding_variance"),
            "recall_at_k":          data.get("recall_at_k"),
            "mean_knn_dist":        _mean_knn_dist(data.get("knn")),
        })
    return rows


def print_best(df: pd.DataFrame) -> None:
    metrics = {
        "intra_similarity":      "max",
        "sequential_similarity": "max",
        "anisotropy":            "min",  # niższe = bardziej izotropowe
        "embedding_variance":    "max",
        "recall_at_k":           "max",
        "mean_knn_dist":         "min",  # bliżsi sąsiedzi = gęstsze klastry
    }
    print("\n=== Najlepsza konfiguracja na metrykę ===")
    for col, direction in metrics.items():
        if df[col].notna().any():
            idx = df[col].idxmax() if direction == "max" else df[col].idxmin()
            row = df.loc[idx]
            print(f"  {col:22s}: {row[col]:.4f}  ->  {row['experiment']} / {row['chunk_config']}")


def main() -> None:
    rows = collect_results(RESULTS_ROOT)
    if not rows:
        print(f"[WARN] Brak plików results.json w {RESULTS_ROOT}")
        return

    df = pd.DataFrame(rows).sort_values("intra_similarity", ascending=False).reset_index(drop=True)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n=== Embedding Quality — porównanie wszystkich konfiguracji ===\n")
    print(df.to_string(index=True))

    print_best(df)

    out_path = RESULTS_ROOT / "comparison.csv"
    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n[OK] Zapisano tabelę do {out_path}")


if __name__ == "__main__":
    main()
