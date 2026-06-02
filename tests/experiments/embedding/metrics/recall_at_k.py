import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_recall_at_k(embeddings: np.ndarray, doc_ids: list, k: int = 5) -> float:
    n = len(embeddings)
    k_actual = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_actual, metric='cosine')
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    k_used = k_actual - 1
    scores = []
    for i, neighbors in enumerate(indices):
        neighbors = neighbors[1:]
        same_doc = sum(1 for j in neighbors if doc_ids[j] == doc_ids[i])
        scores.append(same_doc / k_used)

    return float(np.mean(scores))
