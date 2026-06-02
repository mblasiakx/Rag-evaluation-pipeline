import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RESULTS_FILE = Path(__file__).resolve().parent / "retrieval_experiments_results" / "retrieval_results.json"


def collect_results(results_file: Path) -> list[dict]:
    with open(results_file, encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)
    grouped = (
        df.groupby(["experiment", "chunk_config"])
        .agg(
            precision_at_k=("precision_at_k", "mean"),
            hit_rate=("hit_rate", "mean"),
            mrr=("mrr", "mean"),
            num_queries=("query_text", "count"),
        )
        .reset_index()
    )
    return grouped


def print_best(df: pd.DataFrame) -> None:
    metrics = {
        "precision_at_k": "max",
        "hit_rate":        "max",
        "mrr":             "max",
    }
    print("\n=== Najlepsza konfiguracja na metryke ===")
    for col, direction in metrics.items():
        if df[col].notna().any():
            idx = df[col].idxmax() if direction == "max" else df[col].idxmin()
            row = df.loc[idx]
            print(f"  {col:16s}: {row[col]:.4f}  ->  {row['experiment']} / {row['chunk_config']}")


def main() -> None:
    if not RESULTS_FILE.exists():
        print(f"[ERROR] Brak pliku wynikow: {RESULTS_FILE}")
        print("Uruchom najpierw run_retrieval_eval.py")
        return

    df = collect_results(RESULTS_FILE)
    df = df.sort_values("mrr", ascending=False).reset_index(drop=True)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n=== Retrieval Quality — porownanie wszystkich konfiguracji ===\n")
    print(df.to_string(index=True))

    print_best(df)

    out_path = RESULTS_FILE.parent / "comparison.csv"
    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n[OK] Zapisano tabele do {out_path}")


if __name__ == "__main__":
    main()
