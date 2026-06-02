from sentence_transformers import util
from .model_registry import get_model

def topic_drift(chunks: list[str], query: str) -> float:
    """Średnia odległość chunków od query. Im wyższa, tym chunki bardziej odbiegają od tematu."""
    if not query:
        raise ValueError("topic_drift wymaga niepustego query")
    
    if not chunks:
        return 0.0

    query_emb = get_model().encode(query, convert_to_tensor=True)
    scores = []

    for chunk in chunks:
        chunk_emb = get_model().encode(chunk, convert_to_tensor=True)
        sim = util.cos_sim(query_emb, chunk_emb).item()
        drift = 1 - sim
        scores.append(drift)

    return sum(scores) / len(scores)