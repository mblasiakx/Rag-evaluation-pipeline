import pytest


def  test_retriever_returns_k_results(config,retriever):
      k = config["retrieval"]["k"]
      results = retriever.invoke("test query")
      assert len(results) == k

def test_retrieved_docs_are_non_empty_strings(retriever):
      results = retriever.invoke("test query")
      for doc in results:
          assert isinstance(doc.page_content, str)
          assert len(doc.page_content.strip()) > 0