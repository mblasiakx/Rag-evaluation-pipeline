import os
import pdfplumber
from docx import Document


def load_document(path: str, encoding: str = "utf-8") -> str:
    """Convenience wrapper — loads any supported file and returns raw text."""
    return DocumentLoader(encoding=encoding).load(path)


class DocumentLoader:
    """
    Generic document loader for RAG ingestion.
    Used by rag_experiments via config.
    """

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def load(self, path: str) -> str:
        ext = os.path.splitext(path)[-1].lower()

        if ext == ".txt":
            return self._load_text(path)
        elif ext == ".pdf":
            return self._load_pdf(path)
        elif ext == ".docx":
            return self._load_docx(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _load_text(self, path: str) -> str:
        with open(path, "r", encoding=self.encoding) as f:
            return f.read()

    def _load_pdf(self, path: str) -> str:
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def _load_docx(self, path: str) -> str:
        doc = Document(path)
        paragraphs = [
            p.text.strip()
            for p in doc.paragraphs
            if p.text.strip()
        ]
        return "\n".join(paragraphs)