from .hit import is_hit


def compute_mrr(retrieved_chunks: list, query_offsets: list) -> float:
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if is_hit(query_offsets, chunk):
            return 1.0 / rank
    return 0.0
