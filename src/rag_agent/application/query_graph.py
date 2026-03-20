from typing import TypedDict

from langgraph.graph import END, StateGraph

from rag_agent.domain.models import Chunk, QueryResult
from rag_agent.domain.ports import LLMProvider, Reranker, Retriever


class QueryState(TypedDict):
    question: str
    original_question: str
    sub_queries: list[str]
    chunks: list[Chunk]
    answer: str
    context_quality: str  # "good" or "poor"
    attempts: int
    max_attempts: int


EXPAND_PROMPT = """Break the following question into 2-4 simpler sub-queries that would help retrieve relevant documents. If the question is already simple, return it as-is.

Question: {question}

Return one sub-query per line, nothing else.

Sub-queries:
"""

GRADER_PROMPT = """Given the question and retrieved context below, decide if the context is sufficient to answer the question.

Question: {question}

Context:
{context}

Reply with exactly one word: "good" if the context is relevant and sufficient, or "poor" if it is not.

Answer:"""

REPHRASE_PROMPT = """The following question didn't retrieve good context. Rephrase it to improve retrieval.

Original question: {question}

Write a better search query that might retrieve more relevant documents. Output only the rephrased query, nothing else.

Rephrased query:"""

GENERATE_PROMPT = """Answer the question based on the following context. If the context doesn't contain enough information to answer, say so.

Context:
{context}

Question: {question}

Answer:"""


class QueryGraphBuilder:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        reranker: Reranker | None = None,
        max_attempts: int = 2,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._reranker = reranker
        self._max_attempts = max_attempts

    def _expand(self, state: QueryState) -> QueryState:
        prompt = EXPAND_PROMPT.format(question=state["question"])
        raw = self._llm.generate(prompt).strip()
        sub_queries = [line.strip() for line in raw.splitlines() if line.strip()]
        if not sub_queries:
            sub_queries = [state["question"]]
        return {**state, "sub_queries": sub_queries}

    def _retrieve(self, state: QueryState) -> QueryState:
        all_chunks: dict[str, Chunk] = {}
        queries = state["sub_queries"] if state["sub_queries"] else [state["question"]]
        for query in queries:
            chunks = self._retriever.retrieve(query)
            for chunk in chunks:
                all_chunks[chunk.id] = chunk
        merged = list(all_chunks.values())
        if self._reranker:
            merged = self._reranker.rerank(state["original_question"], merged)
        return {**state, "chunks": merged}

    def _grade(self, state: QueryState) -> QueryState:
        context = "\n\n---\n\n".join(c.content for c in state["chunks"])
        prompt = GRADER_PROMPT.format(
            question=state["original_question"], context=context
        )
        quality = self._llm.generate(prompt).strip().lower()
        if quality not in ("good", "poor"):
            quality = "good"
        return {**state, "context_quality": quality}

    def _rephrase(self, state: QueryState) -> QueryState:
        prompt = REPHRASE_PROMPT.format(question=state["question"])
        rephrased = self._llm.generate(prompt).strip()
        return {**state, "question": rephrased, "attempts": state["attempts"] + 1}

    def _generate(self, state: QueryState) -> QueryState:
        context = "\n\n---\n\n".join(c.content for c in state["chunks"])
        prompt = GENERATE_PROMPT.format(
            question=state["original_question"], context=context
        )
        answer = self._llm.generate(prompt)
        return {**state, "answer": answer}

    def _should_retry(self, state: QueryState) -> str:
        if (
            state["context_quality"] == "poor"
            and state["attempts"] < state["max_attempts"]
        ):
            return "rephrase"
        return "generate"

    def build(self):
        graph = StateGraph(QueryState)

        graph.add_node("expand", self._expand)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade", self._grade)
        graph.add_node("rephrase", self._rephrase)
        graph.add_node("generate", self._generate)

        graph.set_entry_point("expand")
        graph.add_edge("expand", "retrieve")
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges(
            "grade",
            self._should_retry,
            {
                "rephrase": "rephrase",
                "generate": "generate",
            },
        )
        graph.add_edge("rephrase", "retrieve")
        graph.add_edge("generate", END)

        return graph.compile()

    def execute(self, question: str) -> QueryResult:
        graph = self.build()
        initial_state: QueryState = {
            "question": question,
            "original_question": question,
            "sub_queries": [],
            "chunks": [],
            "answer": "",
            "context_quality": "",
            "attempts": 0,
            "max_attempts": self._max_attempts,
        }
        result = graph.invoke(initial_state)
        return QueryResult(
            answer=result["answer"],
            chunks=result["chunks"],
            metadata={
                "attempts": result["attempts"],
                "context_quality": result["context_quality"],
            },
        )
