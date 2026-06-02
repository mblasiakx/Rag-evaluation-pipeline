import yaml
import json
from pathlib import Path
import sys
import os
import nltk
from functools import partial

_CHUNKING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_CHUNKING_ROOT)

# projekt root (zawiera folder rag/)
_PROJECT_ROOT = Path(__file__).resolve()
while not (_PROJECT_ROOT / "rag").exists():
    _PROJECT_ROOT = _PROJECT_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from chunkers.fixed_chunking import FixedChunking
from chunkers.semantic_chunking import SemanticChunking
from metrics.semantic_coherence import semantic_coherence
from metrics.topic_drift import topic_drift
from metrics.chunk_entropy import chunk_entropy
from metrics.chunk_size_stats import chunk_size_std
from metrics.chunk_size_stats import chunk_size_mean
from metrics.boundary_quality import boundary_quality
from metrics.inter_chunk_similarity import inter_chunk_similarity




CHUNKERS = {
    "fixed": FixedChunking,
    "semantic": SemanticChunking
}


def load_text(path):
    return Path(path).read_text(encoding="utf-8")



def save_chunks(chunks, output_path):
  
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # tworzy folder jeśli nie istnieje

    with output_path.open('w', encoding='utf-8') as f:
        for chunk in chunks:
            json.dump(chunk, f, ensure_ascii=False)
            f.write('\n')



def run_experiment(config_path):

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    doc_id = Path(config["input_file"]).stem


    query = config.get("query", "")

    METRICS = {
    "coherence": semantic_coherence,
    "topic_drift": partial(topic_drift, query=query),
    "chunk_entropy": chunk_entropy,
    "chunk_size_std" : chunk_size_std,
    "chunk_size_mean" : chunk_size_mean,
    "boundary_quality" : boundary_quality,
    "inter_chunk_similarity" : inter_chunk_similarity
    }

    text = load_text(config["input_file"])
    chunker_type = config["chunker"]
    metrics_to_compute = config["metrics"]
    params = config.get("params", {})




    if chunker_type not in CHUNKERS:
        raise ValueError(f"Unknown chunker '{chunker_type}'. Available: {list(CHUNKERS.keys())}")

    results = {}

    # ustawienia chunków
    if chunker_type == "fixed":
        if "chunk_sizes" in params:
            chunk_sizes = params["chunk_sizes"]
        elif "chunk_size" in params:
            cs = params["chunk_size"]
            chunk_sizes = cs if isinstance(cs, (list, tuple)) else [cs]
        else:
            raise KeyError("Missing 'chunk_size' or 'chunk_sizes' in config params for fixed chunker.")

        if "overlaps" in params:
            overlaps = params["overlaps"]
            overlaps = overlaps if isinstance(overlaps, (list, tuple)) else [overlaps]
        elif "overlap" in params:
            overlaps = [params["overlap"]]
        else:
            overlaps = [0]

    elif chunker_type == "semantic":
        max_chunk_size = params.get("max_chunk_size", params.get("chunk_size", 300))
        chunk_sizes = [max_chunk_size]
        overlaps = [0]  # semantic ignoruje overlap

    # główna pętla eksperymentu
    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            if chunker_type == "fixed":
                chunker = CHUNKERS[chunker_type](chunk_size, overlap)
            elif chunker_type == "semantic":
                threshold = params.get("threshold", 0.5)
                max_chunk_size = params.get("max_chunk_size", chunk_size)
                chunker = CHUNKERS[chunker_type](threshold=threshold, max_chunk_size=max_chunk_size)

            # generowanie chunków
            try:
                raw_chunks = chunker.chunk(text)  
            except Exception as e:
                print(f"[ERROR] chunker {chunker_type} failed for chunk_size={chunk_size}, overlap={overlap}: {e}")
                raw_chunks = []

            chunks_dicts = []
            for i, ch in enumerate(raw_chunks):
                chunks_dicts.append({
                    "chunk_id": f"{chunker_type}_chunk{chunk_size}_{overlap}_{i}",
                    "doc_id": doc_id,
                    "text": ch["text"],
                    "start_char": ch["start_char"],
                    "end_char": ch["end_char"],
                    "chunk_method": chunker_type,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                })

            # debug: pierwsze 3 chunki
            print(f"DEBUG: {chunker_type} chunk_size={chunk_size} overlap={overlap} -> {len(chunks_dicts)} chunks")
            for i, ch in enumerate(chunks_dicts[:3]):
                preview = ch["text"][:200].replace("\n", " ")
                print(f"  chunk[{i}] len={len(ch['text'])} preview={repr(preview)}")
            
            if chunker_type == "fixed":
                output_path = config["output_chunks_path"].format(
                chunk_size=chunk_size,
                overlap=overlap)
            else:
                output_path = config["output_chunks_path"].format(
                max_chunk_size=max_chunk_size,
                threshold=threshold)
            
            save_chunks(chunks_dicts, output_path)
            print(f"Chunks saved: {output_path}")
            
            chunk_texts = [c["text"] for c in chunks_dicts]
            # liczenie metryk
            metric_results = {}
            for metric in metrics_to_compute:
                try:
                    metric_fn = METRICS[metric]
                    metric_results[metric] = metric_fn(chunk_texts)
                except Exception as e:
                    print(f"[ERROR] metric {metric} failed for chunk_size={chunk_size}, overlap={overlap}: {e}")
                    metric_results[metric] = None

            key = f"chunk_size={chunk_size}_overlap={overlap}"
            results[key] = {
                "num_chunks": len(chunk_texts),
                "metrics": metric_results
            }

    # zapis wyników
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{config['experiment_name']}_results.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Results in: {out_file}")

if __name__ == "__main__":
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    if len(sys.argv) < 2:
        print("Usage: python run_single_experiment.py path/to/config.yaml")
        sys.exit(1)
    run_experiment(sys.argv[1])
