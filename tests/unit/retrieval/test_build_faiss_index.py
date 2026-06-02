import faiss
from tests.experiments.retrieval.experiments.build_faiss_index import select_faiss_index


def test_no_normalization_uses_l2():
      index = select_faiss_index("transformer_no_normalization_pooling", dim=384)
      assert isinstance(index, faiss.IndexFlatL2)

def test_normalized_uses_ip():
      index = select_faiss_index("transformer_pooling_mean", dim=384)
      assert isinstance(index, faiss.IndexFlatIP)