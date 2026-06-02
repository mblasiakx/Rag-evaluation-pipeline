from .hit import is_hit
def compute_precision_at_k(retrieved_chunks: list, query_offsets: list, k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for ch in retrieved_chunks if is_hit(query_offsets, ch))
    return hits / k
