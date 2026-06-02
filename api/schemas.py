from pydantic import BaseModel, field_validator


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v


class AskResponse(BaseModel):
    answer: str
    contexts: list[str]
    latency_ms: int
