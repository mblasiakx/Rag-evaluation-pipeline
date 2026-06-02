from rag.chunking.semantic import SemanticChunker


class SemanticChunking(SemanticChunker):
    """Thin compat wrapper — maps legacy `threshold` kwarg to `similarity_threshold`."""

    def __init__(self, threshold: float = 0.5, max_chunk_size: int = 300):
        super().__init__(similarity_threshold=threshold, max_chunk_size=max_chunk_size)


__all__ = ["SemanticChunking"]
