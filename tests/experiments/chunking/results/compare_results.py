import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = Path(__file__).resolve().parent / "outputs"

RESULT_FILES = {
    "chunk_size_search": "chunk_size_search_results.json",
    "overlap_search":    "overlap_search_results.json",
    "semantic_search":   "semantic_search_results.json",
}

METRICS = ["coherence", "topic_drift", "chunk_entropy",
           "chunk_size_mean", "chunk_size_std", "boundary_quality",
           "inter_chunk_similarity"]


def collect_results() -> list[dict]:
    rows = []
    for experiment, filename in RESULT_FILES.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"[WARN] Brak pliku: {path}")
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for config_key, entry in data.items():
            row = {
                "experiment":  experiment,
                "config":      config_key,
                "num_chunks":  entry.get("num_chunks"),
            }
            for metric in METRICS:
                row[metric] = entry.get("metrics", {}).get(metric)
            rows.append(row)

    return rows


def print_best(df: pd.DataFrame) -> None:
    directions = {
        "coherence":              "max",
        "topic_drift":            "min",
        "chunk_entropy":          "min",
        "chunk_size_std":         "min",
        "boundary_quality":       "max",
        "inter_chunk_similarity": "min",
    }
    print("\n=== Najlepsza konfiguracja na metryke ===")
    for col, direction in directions.items():
        if col not in df.columns or df[col].isna().all():
            continue
        idx = df[col].idxmax() if direction == "max" else df[col].idxmin()
        row = df.loc[idx]
        print(f"  {col:25s}: {row[col]:.4f}  ->  {row['experiment']} / {row['config']}")


def main() -> None:
    rows = collect_results()
    if not rows:
        print(f"[ERROR] Brak plikow wynikow w {RESULTS_DIR}")
        return

    df = pd.DataFrame(rows).sort_values(["experiment", "config"]).reset_index(drop=True)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n=== Chunking Quality — porownanie wszystkich konfiguracji ===\n")
    print(df.to_string(index=True))

    print_best(df)

    out_path = RESULTS_DIR / "comparison.csv"
    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n[OK] Zapisano tabele do {out_path}")


if __name__ == "__main__":
    main()
