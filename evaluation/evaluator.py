from ragas.evaluation import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from sklearn.metrics.pairwise import cosine_similarity
from datasets import Dataset
import pandas as pd


def compute_answer_completeness(answer: str, reference: str, embeddings) -> float:
    answer_vec = embeddings.embed_query(answer)
    reference_vec = embeddings.embed_query(reference)
    score = cosine_similarity([answer_vec], [reference_vec])[0][0]
    return round(float(max(0.0, score)), 4)


def evaluate_answers(qa_data, embeddings):
    df = pd.DataFrame(qa_data)
    df = df.rename(columns={
        "retrieved_contexts": "contexts",
        "reference": "ground_truth"
    })

    dataset = Dataset.from_pandas(df)

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )
    results_df = results.to_pandas()

    completeness_scores = [
        compute_answer_completeness(sample["answer"], sample["reference"], embeddings)
        for sample in qa_data
    ]
    results_df["completeness"] = completeness_scores

    return results_df

