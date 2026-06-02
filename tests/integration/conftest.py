import pytest
import os
import sys
from unittest.mock import MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from rag.file_loader import load_document
from scripts.run_pipeline import load_config, build_chunker, load_questions, PROMPT_MAP
from rag.retriever import create_retriever
from rag.pipeline import run_qa

CONFIGS = [
    os.path.join(PROJECT_ROOT, "configs/config_A.yaml"),
    os.path.join(PROJECT_ROOT, "configs/config_B.yaml"),
    os.path.join(PROJECT_ROOT, "configs/config_C.yaml"),
]


@pytest.fixture(scope="module", params=CONFIGS, ids=["config_A", "config_B", "config_C"])
def config(request):
    return load_config(request.param)


@pytest.fixture(scope="module")
def document(config):
    return load_document(config["data"]["files"][0])


@pytest.fixture(scope="module")
def chunks(config, document):
    chunker = build_chunker(config)
    return chunker.chunk(document)


@pytest.fixture(scope="module")
def _retriever_pair(config, chunks):
    emb = config["embedding"]
    return create_retriever(
        chunks,
        k=config["retrieval"]["k"],
        model_name=emb["model"],
        pooling=emb.get("pooling", "mean"),
        normalize=emb.get("normalize", True),
    )


@pytest.fixture(scope="module")
def retriever(_retriever_pair):
    return _retriever_pair[0]


@pytest.fixture(scope="module")
def embeddings(_retriever_pair):
    return _retriever_pair[1]


@pytest.fixture(scope="module")
def llm():
    mock = MagicMock()
    mock.invoke.return_value = "mocked answer"
    return mock


@pytest.fixture(scope="module")
def questions(config):
    return load_questions(config["data"]["questions"])


@pytest.fixture(scope="module")
def prompt_fn(config):
    return PROMPT_MAP[config["prompt"]["style"]]


@pytest.fixture(scope="module")
def qa_data(llm, retriever, prompt_fn, questions):
    return run_qa(llm, retriever, prompt_fn, questions)
