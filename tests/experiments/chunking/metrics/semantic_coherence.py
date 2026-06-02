from sentence_transformers import util
from .model_registry import get_model

# jak embeddingowo zdania sa do siebie podobne w chunku
def semantic_coherence(chunks):
    if not chunks:
        return 0.0

    scores = []

    for chunk in chunks:
        sentences = [s.strip() for s in chunk.split(".") if s.strip()]
        if len(sentences) < 2:
            scores.append(1.0)
            continue

        embeddings = get_model().encode(sentences, convert_to_tensor=True)
        sim_matrix = util.cos_sim(embeddings, embeddings)
        mean_sim = (sim_matrix.sum() - sim_matrix.shape[0]) / (sim_matrix.shape[0]*(sim_matrix.shape[0]-1))
        scores.append(mean_sim.item())

    return sum(scores) / len(scores)