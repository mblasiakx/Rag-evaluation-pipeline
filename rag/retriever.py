from langchain_community.vectorstores import FAISS
from rag.embedder import build_model, SentenceTransformerEmbeddings, DEFAULT_EMBEDDING_MODEL


def create_retriever(
    chunks: list,
    k: int = 6,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    pooling: str = "mean",
    normalize: bool = True,
):
    texts = [c["text"] if isinstance(c, dict) else c for c in chunks]
    embeddings = SentenceTransformerEmbeddings(
        model=build_model(model_name, pooling=pooling),
        normalize=normalize,
    )
    vectorstore = FAISS.from_texts(texts, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever, embeddings
