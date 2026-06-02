from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_intra_similarity(embeddings, doc_ids):  #mierzy spójność semantyczną w obrębie dokumentu
    #  Średnia cosine similarity między chunkami tego samego dokumentu.
    sims = []
    unique_docs = set(doc_ids)

    for doc in unique_docs:
        idx = [i for i, d in enumerate(doc_ids) if d == doc]
        if len(idx) < 2:
            continue  # nie ma sensu liczyć dla pojedynczego chunku

        sims_doc = cosine_similarity(embeddings[idx])
        # bierzemy tylko górną trójkąt bez diagonali
        sims_doc = sims_doc[np.triu_indices_from(sims_doc, k=1)]
        sims.extend(sims_doc)

    return float(np.mean(sims)) if sims else 0.0


