from tests.experiments.embedding.metrics.sequential_similarity import compute_sequential_similarity
from tests.experiments.embedding.metrics.intra_similarity import compute_intra_similarity
from tests.experiments.embedding.metrics.knn_neighbors import compute_knn_neighbors
from tests.experiments.embedding.metrics.embedding_variance import compute_embedding_variance
from tests.experiments.embedding.metrics.recall_at_k import compute_recall_at_k

import numpy as np      
#. Ma edge case'y które już omawialiśmy — pojedynczy chunk, zerowe wektory, dwa identyczne wektory  
#sequential_similarity


def test_sequential_similarity_returns_float():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = compute_sequential_similarity(emb)
    assert isinstance(result, float)

  # dwa identyczne wektory → similarity = 1.0
def test_sequential_similarity_identical_vectors_return_one():
    emb = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert np.isclose(compute_sequential_similarity(emb), 1.0)

  # wektory prostoapdałe → similarity = 0.0
def test_sequential_similarity_orthogonal_vectors_return_zero():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.isclose(compute_sequential_similarity(emb), 0.0)

def test_sequential_similarity_single_chunk_returns_zero():
    emb = np.array([[1.0, 0.0]])
    assert compute_sequential_similarity(emb) == 0.0

#compute_intra_similarity 

def test_compute_intra_similarity_returns_float():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = compute_intra_similarity(emb,doc_ids=["doc1", "doc1"])
    assert isinstance(result, float)

def test_compute_intra_similarity_identical_vectors_return_one():
    emb = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert np.isclose(compute_intra_similarity(emb, doc_ids=["doc1", "doc1"]), 1.0)

def test_compute_intra_similarity_orthogonal_vectors_return_zero():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.isclose(compute_intra_similarity(emb, doc_ids=["doc1", "doc1"]), 0.0)

def test_compute_intra_similarity_single_chunk_returns_zero():
    emb = np.array([[1.0, 0.0]])
    assert compute_intra_similarity(emb, doc_ids=["doc1"]) == 0.0

  # doc1: identyczne wektory → similarity = 1.0                                                                                                                                                            
      # doc2: prostopadłe → similarity = 0.0                                                                                                                                                                   
      # wynik: średnia = 0.5 
def test_compute_intra_similarity_ignores_cross_doc_pairs(): 
    emb = np.array([ 
          [1.0, 0.0],  # doc1 chunk 0
          [1.0, 0.0],  # doc1 chunk 1
          [1.0, 0.0],  # doc2 chunk 0
          [0.0, 1.0],  # doc2 chunk 1
    ])
    doc_ids = ["doc1", "doc1", "doc2", "doc2"]
    result = compute_intra_similarity(emb, doc_ids=doc_ids)
    assert np.isclose(result, 0.5)

    #compute_knn_neighbors

def test_compute_knn_neighbors_identical_vectors_return_zero():
    emb = np.array([[1.0, 0.0], [1.0, 0.0],[1.0, 0.0]])
    distances, indices = compute_knn_neighbors(emb, k=2)
    assert np.all(np.isclose(distances, 0.0))

def test_compute_knn_neighbors_orthogonal_vectors_return_one():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    distances, indices = compute_knn_neighbors(emb, k=2)
    assert np.all(np.isclose(distances, 1.0))

def test_compute_knn_neighbors_output_shape():
      emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
      distances, indices = compute_knn_neighbors(emb, k=2)
      assert distances.shape == (3, 2)
      assert indices.shape == (3, 2)

def test_compute_knn_neighbors_n_less_than_k_does_not_raise():
      emb = np.array([[1.0, 0.0], [0.0, 1.0]])
      distances, indices = compute_knn_neighbors(emb, k=10)
      assert distances.shape[0] == 2

#compute_embedding_variance

def test_compute_embedding_variance_returns_float():
    emb = np.array([[1.0, 0.0], [1.0, 0.0]])
    result = compute_embedding_variance(emb)
    assert isinstance(result, float)

def test_compute_embedding_variance_identical_vectors_return_zero():
    emb = np.array([[1.0, 0.0], [1.0, 0.0]])
    result = compute_embedding_variance(emb)
    assert np.isclose(result, 0.0)   

def test_compute_embedding_variance_positive():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = compute_embedding_variance(emb)
    assert result >= 0.0   

def test_compute_embedding_single_vector_returns_zero():
    emb = np.array([[1.0, 0.0]])
    result = compute_embedding_variance(emb)
    assert np.isclose(result, 0.0) 

#compute_recall_at_k

def test_compute_recall_at_k_returns_float():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = compute_recall_at_k(emb, doc_ids=["doc1", "doc1"], k=1)
    assert isinstance(result, float)

def test_compute_recall_at_k_same_doc_returns_one():
    emb = np.array([[1.0, 0.0], [1.0, 0.1], [1.0, 0.2]])
    result = compute_recall_at_k(emb, doc_ids=["doc1", "doc1", "doc1"], k=2)
    assert np.isclose(result, 1.0)

def test_compute_recall_at_k_n_less_than_k_does_not_raise():                                                                                                                                                 
      emb = np.array([[1.0, 0.0], [0.0, 1.0]])                                                                                                                                                                 
      result = compute_recall_at_k(emb, doc_ids=["doc1", "doc1"], k=10)                                                                                                                                        
      assert isinstance(result, float)