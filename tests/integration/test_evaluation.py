import pytest
import pandas as pd
from unittest.mock import patch
from evaluation.evaluator import evaluate_answers


@pytest.fixture(scope="module")
def results_df(qa_data, embeddings):
    with patch("evaluation.evaluator.evaluate") as mock_eval:
        mock_eval.return_value.to_pandas.return_value = pd.DataFrame({
            "faithfulness": [0.8] * len(qa_data),
            "answer_relevancy": [0.7] * len(qa_data),
            "context_precision": [0.9] * len(qa_data),
            "context_recall": [0.6] * len(qa_data),
        })
        return evaluate_answers(qa_data, embeddings)


def test_results_df_has_required_columns(results_df):
    for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "completeness"]:
        assert col in results_df.columns


def test_completeness_is_in_valid_range(results_df):
    assert all(0.0 <= s <= 1.0 for s in results_df["completeness"])


def test_results_count_matches_questions(results_df, questions):
    assert len(results_df) == len(questions)
