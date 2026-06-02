from typing import List
import numpy as np
from itertools import combinations
from sentence_transformers import SentenceTransformer

# Cache do embedderów – przyspiesza wielokrotne wywołania dla tego samego modelu
_embedder_cache = {}

def semantic_dispersion(responses: List[str], model_name: str = "all-MiniLM-L6-v2") -> float:
    """
    Oblicza średnią odległość kosinusową między embeddingami odpowiedzi.
    
    Parametry:
    - responses: lista stringów – odpowiedzi modelu na to samo pytanie
    - model_name: nazwa modelu embeddingowego (SentenceTransformer)

    Zwraca:
    - float – średnia odległość kosinusowa (im wyższa, tym większa semantyczna różnorodność)
    """
    if len(responses) < 2:
        return 0.0  # brak wariantów, brak różnorodności

    # Inicjalizacja embeddera lub pobranie z cache
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = SentenceTransformer(model_name)

    embedder = _embedder_cache[model_name]

    # Tworzymy embeddingi
    embeddings = embedder.encode(responses, normalize_embeddings=True)

    # Liczymy średnią odległość kosinusową pomiędzy wszystkimi parami
    distances = [
        1 - np.dot(embeddings[i], embeddings[j])
        for i, j in combinations(range(len(embeddings)), 2)
    ]

    return float(np.mean(distances))
