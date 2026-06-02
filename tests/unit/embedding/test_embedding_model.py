import numpy as np
from rag.embedder import build_model


def test_normalize_true_produces_unit_vectors():
    model = build_model(pooling='mean')
    embeddings = model.encode(
        ["hello world", "foo bar"],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_normalize_false_does_not_normalize():
    model = build_model(pooling='mean')
    embeddings = model.encode(
        ["hello world", "foo bar"],
        convert_to_numpy=True,
        normalize_embeddings=False
    )
    norms = np.linalg.norm(embeddings, axis=1)
    assert not np.allclose(norms, 1.0, atol=1e-5)


def test_different_pooling_produces_different_embeddings():
    texts = ["hello world", "foo bar"]
    emb_mean = build_model(pooling='mean').encode(texts, convert_to_numpy=True)
    emb_cls  = build_model(pooling='cls').encode(texts, convert_to_numpy=True)
    emb_max  = build_model(pooling='max').encode(texts, convert_to_numpy=True)
    assert not np.allclose(emb_mean, emb_cls)
    assert not np.allclose(emb_mean, emb_max)


def test_embedding_count_matches_input():
    texts = ['chunk a', 'chunk b', 'chunk c']
    embeddings = build_model(pooling='mean').encode(texts, convert_to_numpy=True)
    assert len(embeddings) == len(texts)


def test_embedding_dimension():
    embeddings = build_model(pooling='mean').encode(["hello world"], convert_to_numpy=True)
    assert embeddings.shape[1] == 384


def test_embeddings_are_numpy_float_array():
    embeddings = build_model(pooling='mean').encode(["hello world"], convert_to_numpy=True)
    assert isinstance(embeddings, np.ndarray)
    assert np.issubdtype(embeddings.dtype, np.floating)


def test_empty_input_returns_empty_array():
    embeddings = build_model(pooling='mean').encode([], convert_to_numpy=True)
    assert len(embeddings) == 0
