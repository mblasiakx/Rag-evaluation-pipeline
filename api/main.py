import time
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.file_loader import load_document
from rag.pipeline import format_contexts
from rag.generator import create_llm
from rag.retriever import create_retriever
from scripts.run_pipeline import load_config, build_chunker, PROMPT_MAP
from api.schemas import AskRequest, AskResponse

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_A.yaml')

pipeline = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config(CONFIG_PATH)

    chunker = build_chunker(cfg)
    all_chunks = []
    for file_path in cfg["data"]["files"]:
        all_chunks.extend(chunker.chunk(load_document(file_path)))

    emb = cfg["embedding"]
    retriever, _ = create_retriever(
        all_chunks,
        k=cfg["retrieval"]["k"],
        model_name=emb["model"],
        pooling=emb.get("pooling", "mean"),
        normalize=emb.get("normalize", True),
    )

    gen = cfg["generator"]
    llm = create_llm(gen["model"], gen.get("temperature", 0.0))

    pipeline["retriever"] = retriever
    pipeline["llm"] = llm
    pipeline["prompt_fn"] = PROMPT_MAP[cfg["prompt"]["style"]]

    yield

    pipeline.clear()


app = FastAPI(title="RAG API", lifespan=lifespan)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    t0 = time.time()

    docs = pipeline["retriever"].invoke(request.question)
    contexts = [doc.page_content for doc in docs]
    formatted = format_contexts(contexts)
    prompt = pipeline["prompt_fn"](request.question, formatted)
    answer = pipeline["llm"].invoke(prompt)

    latency_ms = round((time.time() - t0) * 1000)

    return AskResponse(answer=answer, contexts=contexts, latency_ms=latency_ms)
