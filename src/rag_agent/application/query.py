from rag_agent.domain.models import QueryResult
from rag_agent.domain.ports import LLMProvider, Reranker, Retriever


PROMPT_TEMPLATE = """Answer the question based on the following context. If the context doesn't contain enough information to
answer, say so.

Context:
{context}

Question: {question}

Answer:"""


class QueryUseCase:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        reranker: Reranker | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._reranker = reranker

    def execute(self, question: str) -> QueryResult:
        chunks = self._retriever.retrieve(question)
        if self._reranker:
            chunks = self._reranker.rerank(question, chunks)
        context = "\n\n---\n\n".join(chunk.content for chunk in chunks)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        answer = self._llm.generate(prompt)
        return QueryResult(answer=answer, chunks=chunks)
