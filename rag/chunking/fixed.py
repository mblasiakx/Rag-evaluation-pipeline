from typing import List
from .base import BaseChunker


class FixedChunker(BaseChunker):
    """Character-based fixed-size chunker with configurable overlap."""

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[dict]:
        if not isinstance(text, str):
            text = str(text)

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunks.append({
                "text": text[start:end],
                "start_char": start,
                "end_char": min(end, text_len),
            })
            start = end - self.overlap

        return chunks