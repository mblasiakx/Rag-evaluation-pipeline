import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from .prompt_runtime.prompt_loader import load_prompt
from .prompt_runtime.build_prompt import render_prompt
from .load_retrieval.retrieval_runtime import RetrievalEngine
from .eval.metrics.accuracy import f1_score, batch_accuracy, exact_match, precision, recall
from .eval.metrics.faithfulness import faithfulness, context_utilization_ratio, token_overlap, hallucination_rate, coverage
from tests.experiments.llm_experiments.registry import get_llm

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parents[3]
RETRIEVAL_BASE = BASE_DIR / "tests" / "experiments"/ "retrieval" / "retrieval_experiments_results" / "transformer_pooling_max" / "chunk256_overlap64"
INDEX_PATH = f"{RETRIEVAL_BASE}/faiss.index"
MAPPING_PATH = f"{RETRIEVAL_BASE}/index_mapping.json"
CHUNKS_PATH = "tests/experiments/data/fixed_overlap/chunk256_overlap64.jsonl"

MODEL_NAME = "gemma3:1b"

PROMPTS_DIR = Path("tests/experiments/prompts_experiments/prompts/experiments")
DATASET_PATH = BASE_DIR / "tests" / "experiments"/ "prompts_experiments" / "eval" / "datasets" / "test1_dataset.json"

RESULTS_DIR = Path("tests/experiments/prompts_experiments/results/prompt_eval")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5
# ----------------------------

# 1️⃣ Load dataset
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 3️⃣ Init model
llm = get_llm(MODEL_NAME)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 4️⃣ Init retrieval
retrieval = RetrievalEngine(INDEX_PATH, MAPPING_PATH, CHUNKS_PATH,embedding_model)

# 5️⃣ Load all prompt files
prompt_files = list(PROMPTS_DIR.glob("*.jinja"))

results_all = []

# 6️⃣ Loop po promptach

for prompt_path in prompt_files:
    prompt_name = prompt_path.stem
    prompt_template = load_prompt(prompt_path)

   
    prompt_results = []

    # 7️⃣ Loop po dataset
    for sample in dataset:
        query = sample["query"]

        # retrieve context
        context_chunks = retrieval.retrieve(query, TOP_K)
        context = "\n".join(context_chunks)

        # render prompt
        prompt = render_prompt(prompt_template, query, context)
        
        # generate answer
        answer = llm.generate(prompt)

         # compute metrics
        prompt_results.append({
            "query": query,
            "answer_gt": sample["answer"],
            "answer_pred": answer,
            "f1": f1_score(answer, sample["answer"]),
            "precision": precision(answer, sample["answer"]),
            "recall": recall(answer, sample["answer"]),
            "faithfulness": faithfulness(answer, context),
            "hallucination_rate" : hallucination_rate(answer, context),
            "coverage" : coverage(answer, context)
        
        })

    # --- SAVE RESULTS ---
    output_path = RESULTS_DIR / f"{prompt_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "prompt": prompt_name,
                "model": MODEL_NAME,
                "top_k": TOP_K,
                "num_samples": len(prompt_results),
                "results": prompt_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[SAVED] {output_path}")

