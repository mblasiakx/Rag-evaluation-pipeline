from sentence_transformers import SentenceTransformer, models
from langchain_core.embeddings import Embeddings

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    pooling: str = "mean",
) -> SentenceTransformer:
    word_embedding = models.Transformer(model_name)
    pooling_layer = models.Pooling(
        word_embedding.get_word_embedding_dimension(),
        pooling_mode=pooling,
    )
    return SentenceTransformer(modules=[word_embedding, pooling_layer])


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model: SentenceTransformer, normalize: bool = True):
        self.model = model
        self.normalize = normalize

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=self.normalize).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=self.normalize)[0].tolist()
