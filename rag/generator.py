from langchain_ollama import OllamaLLM


def create_llm(model_name: str, temperature: float = 0.0) -> OllamaLLM:
    return OllamaLLM(model=model_name, temperature=temperature)


