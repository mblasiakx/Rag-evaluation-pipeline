from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import PunktSentenceTokenizer
from .base import BaseChunker


class SemanticChunker(BaseChunker):
    """
    Sentence-level semantic chunking using embeddings similarity.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        max_chunk_size: int = 300
    ):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size

        self.model = SentenceTransformer(model_name)
        self.tokenizer = PunktSentenceTokenizer()

    def chunk(self, text: str) -> List[Dict]:
        if not isinstance(text, str):
            text = str(text)

        spans = list(self.tokenizer.span_tokenize(text))
        if not spans:
            return []

        chunks = []

        cur_start, cur_end = spans[0]
        cur_text = text[cur_start:cur_end]
        cur_emb = self.model.encode(cur_text)

        for start, end in spans[1:]:
            sent_text = text[start:end]
            sent_emb = self.model.encode(sent_text)

            similarity = util.cos_sim(sent_emb, cur_emb).item()
            new_length = len(cur_text) + len(sent_text)

            if similarity < self.similarity_threshold or new_length > self.max_chunk_size:
                chunks.append({
                    "text": cur_text,
                    "start_char": cur_start,
                    "end_char": cur_end
                })

                cur_start, cur_end = start, end
                cur_text = sent_text
                cur_emb = sent_emb
            else:
                cur_end = end
                cur_text = text[cur_start:cur_end]
                cur_emb = sent_emb

        chunks.append({
            "text": cur_text,
            "start_char": cur_start,
            "end_char": cur_end
        })

        return chunks
