import sys
from pathlib import Path

# --- Dodaj ścieżkę projektu do sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # llm_experiments python -m experients.prompt_obedience.run
sys.path.insert(0, str(PROJECT_ROOT))
from base import BaseLLM
from models.llm_local import LocalLLM
from model_loader import get_llm
# --- Teraz importy lokalnych modułów ---
import json
import csv
import yaml
from model_loader import get_llm
from metrics.semantic_dispersion import semantic_dispersion
from metrics.instruction_compliance_rate import instruction_compliance_rate
from metrics.instruction_features_compliance import instruction_features_compliance
from prompts_experiments.prompt_runtime.prompt_loader import load_prompt
# =========================
# Context resolution logic
# =========================
def resolve_context(sample: dict, config: dict) -> str:
    """
    Resolves context depending on experiment mode.

    mode 1: no context
    mode 2: optional context (model may use own knowledge)
    mode 3: strict context-only (RAG-like)
    """
    mode = config.get("mode", 1)

    if mode == 1:
        return ""

    if mode in (2, 3):
        return sample.get("context", "")

    raise ValueError(f"Unknown mode: {mode}")


# =========================
# Load config
# =========================
CONFIG_PATH = Path("experiments/Prompt_obedience/exp6_mistral_mode3.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


# =========================
# Setup model
# =========================
llm = get_llm(config["model"])
model_name = config["model"]["name"]


# =========================
# Load dataset
# =========================
dataset_path = (PROJECT_ROOT / config["dataset"]["path"]).resolve()
with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

question_field = config["dataset"].get("question_field", "query")
answer_field = config["dataset"].get("answer_field")  # optional


# =========================
# Load prompt templates
# =========================
prompts_dir = (PROJECT_ROOT / config["prompts"]["path"]).resolve()
prompt_files = sorted(prompts_dir.glob("*.jinja"))

# =========================
# Prepare results
# =========================
results_dir = (PROJECT_ROOT / config["results"]["output_dir"]).resolve()
results_dir.mkdir(parents=True, exist_ok=True)


# =========================
# Main experiment loop
# =========================
for prompt_path in prompt_files:
    prompt_name = prompt_path.stem
    template = load_prompt(prompt_path)

    all_results = []

    for sample in dataset:
        query = sample[question_field]
        expected_answer = sample.get(answer_field)

        context_text = resolve_context(sample, config)

        prompt_text = template.render(
            query=query,
            context=context_text
        )

        answer = llm.generate(prompt_text)

        result_row = {
            "query": query,
            "context_used": bool(context_text),
            "answer_pred": answer,
            "answer_gt": expected_answer,
            "model": model_name,
            "prompt": prompt_name,
            "mode": config["mode"],
        }

        # =========================
        # Metrics
        # =========================
        if "semantic_dispersion" in config["metrics"]:
            # single response → dispersion = 0 by definition
            result_row["semantic_dispersion"] = 0.0

        if "instruction_compliance_rate" in config["metrics"]:
            result_row["instruction_compliance_rate"] = instruction_compliance_rate(
                responses=[answer],
                instruction_checks=[[]]  # placeholder for Tryb 1
            )

        if "instruction_features_compliance" in config["metrics"]:
            result_row["instruction_features_compliance"] = instruction_features_compliance(
                response=answer,
                instruction_checks=[]
            )

        all_results.append(result_row)


    if not all_results:
        print(f"[WARN] No results for prompt {prompt_name}, skipping save")
        continue

    # =========================
    # Save JSON
    # =========================
    if config["results"].get("save_json", True):
        json_path = results_dir / f"{prompt_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment": config["experiment_name"],
                    "prompt": prompt_name,
                    "model": model_name,
                    "mode": config["mode"],
                    "num_samples": len(all_results),
                    "results": all_results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[SAVED JSON] {json_path}")

    # =========================
    # Save CSV
    # =========================
"""  if config["results"].get("save_csv", False):
        csv_path = results_dir / f"{prompt_name}.csv"

        fieldnames = list(all_results[0].keys())

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        print(f"[SAVED CSV] {csv_path}")"""