import numpy as np                                                                                                 
                                                                                            
def compute_sequential_similarity(embeddings: np.ndarray) -> float:
    """
    Średnia cosine similarity między kolejnymi chunkami (chunk_i vs chunk_i+1).
    Wyższa wartość niż intra_similarity oznacza, że embeddingi odzwierciedlają
    lokalny przepływ tekstu, a nie tylko globalną tematykę dokumentu.
    """
    if len(embeddings) < 2:
        return 0.0

    sims = []
    for i in range(len(embeddings) - 1):
        a, b = embeddings[i], embeddings[i + 1]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            continue
        sims.append(float(np.dot(a, b) / (norm_a * norm_b)))

    return float(np.mean(sims)) if sims else 0.0
