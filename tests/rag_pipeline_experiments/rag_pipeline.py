class RAGPipeline:
    def __init__(self, chunker, embedder, retriever, prompt, llm):
        self.chunker = chunker
        self.embedder = embedder
        self.retriever = retriever
        self.prompt = prompt
        self.llm = llm

    def run(self, question):
        chunks = self.chunker.chunk(question)
        embeddings = self.embedder.embed(chunks)
        contexts, retrieval_meta = self.retriever.retrieve(embeddings)
        prompt_text = self.prompt.format(question, contexts)
        answer = self.llm.generate(prompt_text)
        return answer, contexts, retrieval_meta