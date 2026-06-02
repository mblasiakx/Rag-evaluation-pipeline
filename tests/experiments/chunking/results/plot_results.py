from pathlib import Path
import matplotlib.pyplot as plt
import json

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

METRICS = ["coherence", "topic_drift", "chunk_entropy", "chunk_size_mean", "boundary_quality", "inter_chunk_similarity"]


def load_results(filename: str) -> dict:
    path = OUTPUTS_DIR / filename
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_chunk_size_search():
    data = load_results("chunk_size_search_results.json")

    chunk_sizes = []
    metric_values = {m: [] for m in METRICS}

    for key, value in sorted(data.items(), key=lambda x: int(x[0].split("=")[1].split("_")[0])):
        chunk_sizes.append(int(key.split("=")[1].split("_")[0]))
        for m in METRICS:
            metric_values[m].append(value["metrics"].get(m))

    fig, ax = plt.subplots(figsize=(10, 5))
    for m in METRICS:
        vals = metric_values[m]
        if any(v is not None for v in vals):
            ax.plot(chunk_sizes, vals, marker="o", label=m)

    ax.set_xlabel("chunk_size")
    ax.set_ylabel("metric value")
    ax.set_title("Metrics vs chunk_size (overlap=20)")
    ax.legend()
    fig.tight_layout()
    out = OUTPUTS_DIR / "metrics_vs_chunk_size.png"
    fig.savefig(out)
    print(f"[OK] Saved {out}")
    plt.show()


if __name__ == "__main__":
    plot_chunk_size_search()
