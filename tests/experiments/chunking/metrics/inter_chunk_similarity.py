from sentence_transformers import util
from .model_registry import get_model

def inter_chunk_similarity(chunks: list[str]) -> float:
    if len(chunks) < 2:
        return 0.0
    
    scores = []
    for i in range(len(chunks) - 1):
        emb_a = get_model().encode(chunks[i], convert_to_tensor=True)
        emb_b = get_model().encode(chunks[i+1], convert_to_tensor=True)
        sim = util.cos_sim(emb_a, emb_b).item()
        scores.append(sim)
    
    return sum(scores) / len(scores)