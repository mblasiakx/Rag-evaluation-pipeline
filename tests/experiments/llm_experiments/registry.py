from .models.llm_local import LocalLLM


def get_llm(model_name: str):
    """
    Zwraca instancję LLM na podstawie nazwy.
    """

    # -------- LOCAL MODELS --------
    if model_name.startswith("gemma"):
        return LocalLLM(
            model_name=model_name,
            temperature=0.0,
            max_tokens=512,
        )

    if model_name.startswith("mistral"):
        return LocalLLM(
            model_name=model_name,
            temperature=0.0,
            max_tokens=512,
        )

    ## -------- OPENAI --------
    #if model_name.startswith("gpt-"):
    #    return OpenAILLM(model_name=model_name)

    raise ValueError(f"Unknown model: {model_name}")