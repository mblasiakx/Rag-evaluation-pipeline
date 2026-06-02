from sklearn.cluster import KMeans
import math
from .model_registry import get_model


def _entropy_single(chunk: str, n_clusters: int = 3) -> float:
    """Jak bardzo wymieszane tematy są w pojedynczym chunku (0=jednorodny, 1=max mieszanka)."""
    sentences = [s.strip() for s in chunk.split(".") if len(s.strip()) > 0]
    if len(sentences) < 2:
        return 0.0

    embeddings = get_model().encode(sentences, convert_to_tensor=True)
    k = min(n_clusters, len(sentences))
    kmeans = KMeans(n_clusters=k, n_init="auto")
    labels = kmeans.fit_predict(embeddings.cpu().numpy())

    total = len(labels)
    counts = [list(labels).count(i) for i in range(k)]

    entropy = 0.0
    for c in counts:
        p = c / total
        if p > 0:
            entropy -= p * math.log(p, 2)

    max_entropy = math.log(k, 2)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def chunk_entropy(chunks: list[str], n_clusters: int = 3) -> float:
    """Średnia entropia po wszystkich chunkach. Interfejs: List[str] -> float."""
    if not chunks:
        return 0.0
    scores = [_entropy_single(chunk, n_clusters) for chunk in chunks]
    return sum(scores) / len(scores)