
def format_zero_shot(question: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}"

def format_one_shot(question: str, context: str) -> str:
    example = (
        "Q: What is the Champions League?\n"
        "A: Champions League is a football tournament for clubs from many countries.\n\n"
    )
    return f"Context:\n{context}\n\n{example}Q: {question}\nA:"

def format_few_shot(question: str, context: str) -> str:
    examples = (
        "Q: What is the Champions League?\n"
        "A: Champions League is a football tournament for clubs from many countries.\n\n"
        "Q: How often is the Champions League played?\n"
        "A: Champions League is played every year.\n\n"
    )
    return f"Context:\n{context}\n\n{examples}Q: {question}\nA:"

def format_cot(question: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\nLet's think step by step."

def format_prompt_injection(question: str, context: str) -> str:
    return (
        f"You are a football analyst. Use the context below to answer the question "
        f"truthfully and precisely.\n\nContext:\n{context}\n\nQuestion: {question}"
    )