import json
import faiss
import numpy as np
from typing import List
from pathlib import Path

class RetrievalEngine:
    """
    Runtime retrieval dla eksperymentów promptów.
    Odpowiada WYŁĄCZNIE za:
    - wczytanie FAISS
    - mapowanie faiss_id -> chunk_id
    - zwrócenie tekstów chunków
    """

    def __init__(
        self,
        index_path: str,
        mapping_path: str,
        chunks_path: str,
        embed_model,
    ):
        """
        index_path: ścieżka do faiss.index
        mapping_path: ścieżka do index_mapping.json
        chunks_path: ścieżka do chunks.json
        embed_model: model embeddingów (musi mieć .encode)
        """

        # --- FAISS ---
        self.index = faiss.read_index(index_path)

        # --- Mapping FAISS -> chunk_id ---
        with open(mapping_path, "r", encoding="utf-8") as f:
            self.faiss_id_to_chunk_id = json.load(f)["faiss_id_to_chunk_id"]

        # --- Chunki (źródło tekstu) ---
        self.chunks = {}
        chunks_path = Path(chunks_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.chunks[obj["chunk_id"]] = obj["text"]


        self.embed_model = embed_model

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Zwraca listę tekstów chunków (bez metadanych)
        """

        # --- Embed zapytania ---
        query_vector = self.embed_model.encode(
            [query],
            normalize_embeddings=True
        )
        query_vector = np.asarray(query_vector, dtype="float32")

        # --- FAISS search ---
        distances, indices = self.index.search(query_vector, top_k)

        # --- FAISS id -> chunk_id -> text ---
        retrieved_texts = []

        for faiss_id in indices[0]:
            chunk_id = self.faiss_id_to_chunk_id.get(str(faiss_id))
            if not chunk_id:
                continue

            text = self.chunks.get(chunk_id)
            if text:
                retrieved_texts.append(text)

        return retrieved_texts