from tests.experiments.embedding.experiments.embedding_quality.evaluate_single_chunk_file import evaluate_single_chunk_file
import pytest
import json


@pytest.fixture(scope="module")
def sample_chunk_file(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("chunks")
    chunk_file = tmp_path / "chunks.jsonl"
    chunk_file.write_text(
        '{"chunk_id": "c0", "doc_id": "doc1", "text": "hello world", "start_char": 0, "end_char": 11}\n'
        '{"chunk_id": "c1", "doc_id": "doc1", "text": "foo bar baz", "start_char": 11, "end_char": 22}\n'
    )
    return chunk_file


@pytest.fixture(scope="module")
def base_config():
    return {
        "model": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "pooling": "mean",
            "normalize": True,
            "device": "cpu",
            "batch_size": 32
        },
        "metrics": {
            "compute_intra_similarity": True,
            "compute_knn_neighbors": False,
            "compute_anisotropy": False,
            "compute_embedding_variance": False,
            "compute_recall_at_k": False,
            "compute_sequential_similarity": False
        },
        "outputs": {
            "save_embeddings": True,
            "save_similarity_matrix": False
        }
    }


@pytest.fixture(scope="module")
def results_json(sample_chunk_file, base_config, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("results")
    evaluate_single_chunk_file(sample_chunk_file, tmp_path, base_config)
    return json.load(open(tmp_path / "results.json"))


@pytest.fixture(scope="module")
def output_dir(sample_chunk_file, base_config, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("outputs")
    evaluate_single_chunk_file(sample_chunk_file, tmp_path, base_config)
    return tmp_path


# --- pliki wyjściowe ---

@pytest.mark.slow
def test_files_are_created(output_dir):
    assert (output_dir / "embeddings.npy").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "results.json").exists()


# --- metadata.json ---

@pytest.mark.slow
def test_metadata_json_structure(output_dir):
    meta = json.load(open(output_dir / "metadata.json"))
    assert "chunks" in meta
    for chunk in meta["chunks"]:
        assert {"chunk_id", "doc_id", "start_char", "end_char"}.issubset(chunk.keys())
        assert isinstance(chunk["chunk_id"], str)
        assert isinstance(chunk["doc_id"], str)
        assert isinstance(chunk["start_char"], int)
        assert isinstance(chunk["end_char"], int)
        assert chunk["start_char"] < chunk["end_char"]


# --- results.json ---

@pytest.mark.slow
def test_results_json_has_required_keys(results_json):
    assert "intra_similarity" in results_json


@pytest.mark.slow
def test_results_json_scalar_metrics_are_numeric(results_json):
    assert isinstance(results_json["intra_similarity"], float)


@pytest.mark.slow
def test_results_json_intra_similarity_in_valid_range(results_json):
    assert -1.0 <= results_json["intra_similarity"] <= 1.0
