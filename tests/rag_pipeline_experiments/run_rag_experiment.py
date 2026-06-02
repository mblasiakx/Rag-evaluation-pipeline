import random
import importlib
import json
import yaml
from pathlib import Path
from rag_pipeline import RAGPipeline


# ============================================================
# Utils
# ============================================================

def import_class(path: str):
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_yaml(path: str, base_dir: Path | None = None) -> dict:
    path = Path(path)
    if not path.is_absolute() and base_dir:
        path = base_dir / path

    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"YAML config is empty: {path}")

    return data


# ============================================================
# Registries (mapy klas)
# ============================================================

CHUNKER_MAP = {
    "semantic": "chunking.chunkers.semantic_chunking.SemanticChunker",
    "fixed": "chunking.chunkers.fixed_chunking.FixedChunking",
}

EMBEDDER_MAP = {
    "text-embedding-3-large": "rag.embedding.text_embedding.TextEmbedder",
}

RETRIEVER_MAP = {
    "hybrid": "rag.retrieval.hybrid.HybridRetriever",
}

LLM_MAP = {
    "openai": "rag.llm.openai.OpenAIModel",
}

PROMPT_MAP = {
    "prompt_template": "rag.prompt.loader.PromptTemplate",
}

METRIC_MAP = {
  "faithfulness": "prompts_experiments.eval.metrcs.faithfulness.faithfulness"
}


# ============================================================
# Metrics builder
# ============================================================

def build_metrics(metrics_cfg: dict):
    metrics = {}

    for metric_name, cfg in metrics_cfg.items():
        cfg = cfg.copy()
        metric_type = cfg.pop("type")

        if metric_type not in METRIC_MAP:
            raise ValueError(f"Unknown metric type: {metric_type}")

        metric_cls = import_class(METRIC_MAP[metric_type])
        metrics[metric_name] = metric_cls(**cfg)

    return metrics


def aggregate_results(results: list[dict]) -> dict:
    summary = {}

    metric_names = results[0]["metrics"].keys()

    for name in metric_names:
        values = [
            r["metrics"][name]
            for r in results
            if isinstance(r["metrics"][name], (int, float))
        ]

        if values:
            summary[name] = {
                "mean": sum(values) / len(values),
                "count": len(values),
            }

    return summary


# ============================================================
# Main experiment runner
# ============================================================

def run_experiment(experiment_cfg_path: str):

    # ----------------------------
    # Load configs
    # ----------------------------
    exp_cfg = load_yaml(experiment_cfg_path)
    rag_cfg = load_yaml(exp_cfg["rag_config_ref"])["rag"]

    random.seed(exp_cfg["execution"]["seed"])

    # ----------------------------
    # Build RAG components
    # ----------------------------

    # Chunking
    chunking_cfg = rag_cfg["chunking"]
    chunker_cls = import_class(CHUNKER_MAP[chunking_cfg["type"]])
    chunker = chunker_cls(**chunking_cfg.get("config", chunking_cfg))

    # Embedding
    embedding_cfg = rag_cfg["embedding"]
    embedder_cls = import_class(EMBEDDER_MAP[embedding_cfg["model"]])
    embedder = embedder_cls(**embedding_cfg)

    # Retrieval
    retrieval_cfg = rag_cfg["retrieval"]
    retriever_cls = import_class(RETRIEVER_MAP[retrieval_cfg["type"]])
    retriever = retriever_cls(embedder=embedder, **retrieval_cfg)

    # Prompt
    prompt_cfg = rag_cfg["prompt"]
    prompt_cls = import_class(PROMPT_MAP["prompt_template"])
    prompt = prompt_cls(
        template_ref=prompt_cfg["template_ref"],
        max_context_tokens=prompt_cfg["max_context_tokens"],
    )

    # LLM
    llm_cfg = rag_cfg["llm"]
    llm_cls = import_class(LLM_MAP[llm_cfg["provider"]])
    llm = llm_cls(**llm_cfg)

    # Pipeline
    pipeline = RAGPipeline(
        chunker=chunker,
        embedder=embedder,
        retriever=retriever,
        prompt=prompt,
        llm=llm,
    )

    # ----------------------------
    # Dataset
    # ----------------------------
    dataset_cfg = exp_cfg["dataset"]
    dataset_path = Path(dataset_cfg["path"])

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    # ----------------------------
    # Metrics (FROM CONFIG)
    # ----------------------------
    metrics = build_metrics(exp_cfg["evaluation"]["metrics"])

    # ----------------------------
    # Run experiment
    # ----------------------------
    results = []

    for sample in dataset:
        question = sample[dataset_cfg["input_field"]]
        reference = sample.get(dataset_cfg.get("reference_field"))

        answer, contexts, retrieval_meta = pipeline.run(question)

        sample_metrics = {}
        for name, metric in metrics.items():
            sample_metrics[name] = metric.compute(
                question=question,
                answer=answer,
                contexts=contexts,
                reference=reference,
            )

        results.append({
            "question": question,
            "answer": answer,
            "metrics": sample_metrics,
        })

    summary = aggregate_results(results)

    # ----------------------------
    # Logging
    # ----------------------------
    output_dir = Path(exp_cfg["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(" Eksperyment zakończony")
    print(" Wyniki zapisane w:", output_dir)