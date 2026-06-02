from openai import OpenAI
from llm_experiments.base import BaseLLM

class OpenAILLM(BaseLLM):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()