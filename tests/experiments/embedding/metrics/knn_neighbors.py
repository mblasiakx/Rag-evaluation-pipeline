from sklearn.neighbors import NearestNeighbors

def compute_knn_neighbors(embeddings, ids=None, k=5):
    n = len(embeddings)
    k_actual = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_actual, metric='cosine')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    return distances[:, 1:], indices[:, 1:]