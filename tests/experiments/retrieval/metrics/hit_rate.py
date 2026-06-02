from .hit import is_hit

def compute_hit_rate(retrieved_chunks: list, query_offsets: list) -> float:
    return 1.0 if any(is_hit(query_offsets, ch) for ch in retrieved_chunks) else 0.0
