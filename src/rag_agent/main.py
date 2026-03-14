from dotenv import load_dotenv

from rag_agent.adapters.outbound import (
    FixedSizeChunker,
    MarkdownHeaderChunker,
    SemanticChunker,
    MarkdownDocLoader,
    AzureOpenAIEmbedder,
    AzureOpenAILLM,
    DenseRetriever,
    BM25SparseRetriever,
    ChromaVectorStore,
    PostgresDocumentRepository,
    PostgresChunkRepository,
)
from rag_agent.adapters.inbound.cli import parse_args
from rag_agent.application import IngestUseCase, QueryUseCase
from rag_agent.config import AppConfig

load_dotenv()


def build_config(config_path: str) -> AppConfig:
    return AppConfig.from_yaml(config_path)


def cmd_ingest(config: AppConfig) -> None:
    loader = MarkdownDocLoader(path=config.data_source.path)
    embedder = AzureOpenAIEmbedder(model=config.embedder.model)
    if config.chunker.strategy == "fixed-size":
        chunker = FixedSizeChunker(
            chunk_size=config.chunker.chunk_size,
            chunk_overlap=config.chunker.chunk_overlap,
        )
    elif config.chunker.strategy == "markdown-header":
        chunker = MarkdownHeaderChunker()
    elif config.chunker.strategy == "semantic":
        chunker = SemanticChunker(
            embedder=embedder,
            threshold=config.chunker.threshold,
            min_chunk_size=config.chunker.min_chunk_size,
        )
    store = ChromaVectorStore(
        collection_name=config.vector_store.collection_name,
        path=config.vector_store.path,
    )
    document_repo = None
    chunk_repo = None
    if config.database.enabled:
        document_repo = PostgresDocumentRepository(config.database.url)
        chunk_repo = PostgresChunkRepository(config.database.url)

    use_case = IngestUseCase(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        vector_store=store,
        document_repository=document_repo,
        chunk_repository=chunk_repo,
    )
    chunks = use_case.execute()
    print(f"Ingested {len(chunks)} chunks")

    if embedder.last_usage:
        usage = embedder.last_usage
        print(
            f"Embeddings: {usage['num_texts']} texts, dim={usage['embedding_dim']}, {usage['total_tokens']} tokens"
        )


def cmd_query(config: AppConfig, question: str) -> None:
    embedder = AzureOpenAIEmbedder(model=config.embedder.model)
    store = ChromaVectorStore(
        collection_name=config.vector_store.collection_name,
        path=config.vector_store.path,
    )
    if config.retriever.provider == "dense":
        retriever = DenseRetriever(
            embedder=embedder,
            vector_store=store,
            top_k=config.retriever.top_k,
        )
    elif config.retriever.provider == "bm25_sparse":
        chunk_repo = PostgresChunkRepository(config.database.url)
        retriever = BM25SparseRetriever(
            chunk_repository=chunk_repo,
            top_k=config.retriever.top_k,
        )
    llm = AzureOpenAILLM(
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )

    use_case = QueryUseCase(retriever=retriever, llm=llm)
    result = use_case.execute(question)

    print(f"\nAnswer: {result.answer}\n")
    print("Sources:")
    for chunk in result.chunks:
        source = chunk.metadata.get("source", "unknown")
        print(f"  - {source}")

    print()
    if embedder.last_usage:
        usage = embedder.last_usage
        print(f"Embed: {usage['total_tokens']} tokens")
    if llm.last_usage:
        usage = llm.last_usage
        print(
            f"LLM: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion = {usage['total_tokens']} total tokens ({usage['model']})"
        )


def main() -> None:
    args = parse_args()
    config = build_config(args.config)

    if args.command == "ingest":
        cmd_ingest(config)
    elif args.command == "query":
        cmd_query(config, args.question)


if __name__ == "__main__":
    main()
