import argparse
import yaml
import json
import csv
import sys
import os
import mlflow
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()

from rag.file_loader import load_document
from rag.chunking.fixed import FixedChunker
from rag.chunking.semantic import SemanticChunker
from rag.retriever import create_retriever
from rag.generator import create_llm
from rag.pipeline import run_qa
from evaluation.evaluator import evaluate_answers
from prompts.prompt_styles import (
    format_zero_shot, format_one_shot, format_few_shot,
    format_cot, format_prompt_injection,
)

PROMPT_MAP = {
    "zero_shot": format_zero_shot,
    "one_shot": format_one_shot,
    "few_shot": format_few_shot,
    "chain_of_thought": format_cot,
    "prompt_injection": format_prompt_injection,
}


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_questions(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_chunker(cfg: dict):
    c = cfg["chunking"]
    if c["type"] == "fixed":
        return FixedChunker(c["chunk_size"], c["overlap"])
    if c["type"] == "semantic":
        return SemanticChunker(
            similarity_threshold=c["similarity_threshold"],
            max_chunk_size=c["max_chunk_size"],
        )
    raise ValueError(f"Unknown chunker type: {c['type']}")


def main(config_path: str):
    cfg = load_config(config_path)
    config_name = cfg["name"]
    print(f"Running pipeline: {config_name}")

    # --- documents ---
    chunker = build_chunker(cfg)
    all_chunks = []
    for file_path in cfg["data"]["files"]:
        all_chunks.extend(chunker.chunk(load_document(file_path)))

    # --- questions ---
    questions = load_questions(cfg["data"]["questions"])

    # --- retriever ---
    emb = cfg["embedding"]
    retriever, embeddings = create_retriever(
        all_chunks,
        k=cfg["retrieval"]["k"],
        model_name=emb["model"],
        pooling=emb.get("pooling", "mean"),
        normalize=emb.get("normalize", True),
    )

    # --- llm ---
    gen = cfg["generator"]
    llm = create_llm(gen["model"], gen.get("temperature", 0.0))

    # --- prompt ---
    prompt_style = cfg["prompt"]["style"]
    prompt_fn = PROMPT_MAP[prompt_style]

    # --- run ---
    qa_data = run_qa(llm, retriever, prompt_fn, questions)

    # --- evaluate ---
    results_df = evaluate_answers(qa_data, embeddings)
    results_df["formatted_contexts"] = [s["formatted_contexts"] for s in qa_data]
    results_df["latency_ms"] = [s["latency_ms"] for s in qa_data]

    # --- save ---
    os.makedirs("evaluation/results", exist_ok=True)
    csv_path = f"evaluation/results/{config_name}_results.csv"
    results_df.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")
    print(f"Results saved to {csv_path}")

    # --- mlflow ---
    mlflow.set_experiment("pipeline_comparison")
    with mlflow.start_run(run_name=config_name):
        mlflow.log_param("config_name", config_name)
        mlflow.log_param("chunking_type", cfg["chunking"]["type"])
        c = cfg["chunking"]
        if c["type"] == "fixed":
            mlflow.log_param("chunk_size", c["chunk_size"])
            mlflow.log_param("overlap", c["overlap"])
        elif c["type"] == "semantic":
            mlflow.log_param("similarity_threshold", c["similarity_threshold"])
            mlflow.log_param("max_chunk_size", c["max_chunk_size"])
        mlflow.log_param("embedding_model", emb["model"])
        mlflow.log_param("normalize", emb.get("normalize", True))
        mlflow.log_param("k", cfg["retrieval"]["k"])
        mlflow.log_param("generator_model", gen["model"])
        mlflow.log_param("temperature", gen.get("temperature", 0.0))
        mlflow.log_param("prompt_style", prompt_style)
        mlflow.log_param("num_questions", len(questions))
        mlflow.log_param("num_chunks", len(all_chunks))

        for metric in results_df.columns:
            if results_df[metric].dtype in ["float64", "int64"]:
                mlflow.log_metric(metric, results_df[metric].mean())

        mlflow.log_artifact(csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
