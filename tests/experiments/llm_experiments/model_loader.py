from base import BaseLLM
from models.llm_local import LocalLLM
def get_llm(model_config: dict) -> BaseLLM:
    
    provider = model_config["provider"]
    
    if provider == "local":
        return LocalLLM(
            model_name=model_config["name"],
            base_url=model_config.get("base_url", "http://localhost:11434"),
            temperature=model_config.get("temperature", 0.0),
            max_tokens=model_config.get("max_tokens", 512),
        )


  

    raise ValueError(f"Unknown provider: {provider}")