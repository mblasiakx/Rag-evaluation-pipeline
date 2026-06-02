import sys
import yaml
from pathlib import Path
from evaluate_single_chunk_file import evaluate_single_chunk_file

sys.stdout.reconfigure(encoding="utf-8")


def run_experiment(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    chunk_dirs = config["paths"]["chunks_dirs"]
    save_root = Path(config["paths"]["save_root"])

    print(f"[INFO] Running experiment: {config['experiment_name']}")

    all_chunk_files = []
    for chunk_dir in chunk_dirs:
        chunk_dir = Path(chunk_dir)
        chunk_files = list(chunk_dir.glob("*.jsonl"))

        if not chunk_files:
            print(f"[WARN] No chunk files found in {chunk_dir}")
            continue

        print(f"\n[INFO] Found {len(chunk_files)} chunk files in: {chunk_dir}")
        all_chunk_files.extend(chunk_files)

    for chunk_file in all_chunk_files:
        out_dir = save_root / chunk_file.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[RUN] Evaluating chunks: {chunk_file}")
        print(f"[OUT] Saving results to: {out_dir}")

        evaluate_single_chunk_file(
            chunk_file=chunk_file,
            out_dir=out_dir,
            config=config,
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_embedding_quality.py config.yaml")
        sys.exit(1)
    run_experiment(sys.argv[1])
