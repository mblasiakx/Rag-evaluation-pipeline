import pytest
from pathlib import Path
import os
from rag.file_loader import load_document

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
@pytest.fixture(params=["Lyon.txt", "Lyon.pdf", "Lyon.docx"])

def filepath(request):
    return DATA_DIR / request.param

def test_load_document_exists(filepath) : 
    assert filepath.exists()

def test_load_document_supported_formats(filepath) : 
    content = load_document(str(filepath))
    assert isinstance(content, str) 

def test_load_document_not_empty(filepath) : 
    content = load_document(str(filepath))
    assert len(content.strip()) > 0


