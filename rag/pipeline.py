import time


def format_contexts(contexts: list[str]) -> str:
    return "\n".join([f"[{i+1}] {c.strip()}" for i, c in enumerate(contexts)])


def run_qa(llm, retriever, prompt_fn, questions):
    qa_data = []
    for q in questions:
        t0 = time.time()
        docs = retriever.invoke(q["question"])
        raw_contexts = [doc.page_content for doc in docs]
        formatted_contexts = format_contexts(raw_contexts)
        full_prompt = prompt_fn(q["question"], formatted_contexts)
        answer = llm.invoke(full_prompt)
        latency_ms = round((time.time() - t0) * 1000)

        qa_data.append({
            "question": q["question"],
            "answer": answer,
            "retrieved_contexts": raw_contexts,
            "formatted_contexts": formatted_contexts,
            "reference": q["reference"],
            "latency_ms": latency_ms,
        })
    return qa_data
