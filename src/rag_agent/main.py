from dotenv import load_dotenv

from rag_agent.adapters.inbound.cli import parse_args
from rag_agent.adapters.outbound import (
    AzureOpenAIEmbedder,
    AzureOpenAILLM,
    BM25SparseRetriever,
    ChromaVectorStore,
    CohereReranker,
    CrossEncoderReranker,
    DenseRetriever,
    FixedSizeChunker,
    HybridRetriever,
    MarkdownDocLoader,
    MarkdownHeaderChunker,
    PdfDocLoader,
    PostgresChunkRepository,
    PostgresDocumentRepository,
    SemanticChunker,
)
from rag_agent.application import IngestUseCase, QueryGraphBuilder
from rag_agent.config import AppConfig

load_dotenv()


def build_loaders(config: AppConfig):
    loader_map = {
        "markdown": MarkdownDocLoader,
        "pdf": PdfDocLoader,
    }
    return [
        loader_map[loader](path=path)
        for path in config.data_source.paths
        for loader in config.data_source.types
    ]


def build_embedder(config: AppConfig):
    return AzureOpenAIEmbedder(model=config.embedder.model)


def build_chunker(config: AppConfig, embedder):
    if config.chunker.strategy == "fixed-size":
        return FixedSizeChunker(
            chunk_size=config.chunker.chunk_size,
            chunk_overlap=config.chunker.chunk_overlap,
        )
    elif config.chunker.strategy == "markdown-header":
        return MarkdownHeaderChunker()
    elif config.chunker.strategy == "semantic":
        return SemanticChunker(
            embedder=embedder,
            threshold=config.chunker.threshold,
            min_chunk_size=config.chunker.min_chunk_size,
        )


def build_vector_store(config: AppConfig):
    return ChromaVectorStore(
        collection_name=config.vector_store.collection_name,
        path=config.vector_store.path,
    )


def build_repos(config: AppConfig):
    if not config.database.enabled:
        return None, None
    return (
        PostgresDocumentRepository(config.database.url),
        PostgresChunkRepository(config.database.url),
    )


def build_retriever(config: AppConfig, embedder, store):
    if config.retriever.provider == "dense":
        return DenseRetriever(
            embedder=embedder,
            vector_store=store,
            top_k=config.retriever.top_k,
        )
    elif config.retriever.provider == "bm25_sparse":
        chunk_repo = PostgresChunkRepository(config.database.url)
        return BM25SparseRetriever(
            chunk_repository=chunk_repo,
            top_k=config.retriever.top_k,
        )
    elif config.retriever.provider == "hybrid":
        chunk_repo = PostgresChunkRepository(config.database.url)
        dense = DenseRetriever(
            embedder=embedder,
            vector_store=store,
            top_k=config.retriever.top_k,
        )
        sparse = BM25SparseRetriever(
            chunk_repository=chunk_repo,
            top_k=config.retriever.top_k,
        )
        return HybridRetriever(
            dense_retriever=dense,
            sparse_retriever=sparse,
            top_k=config.retriever.top_k,
        )


def build_llm(config: AppConfig):
    return AzureOpenAILLM(
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )


def build_reranker(config: AppConfig):
    if config.reranker.provider == "cohere":
        return CohereReranker(top_k=config.reranker.top_k)
    elif config.reranker.provider == "cross_encoder":
        return CrossEncoderReranker(top_k=config.reranker.top_k)
    return None


def cmd_ingest(config: AppConfig) -> None:
    embedder = build_embedder(config)
    loaders = build_loaders(config)
    chunker = build_chunker(config, embedder)
    store = build_vector_store(config)
    document_repo, chunk_repo = build_repos(config)

    use_case = IngestUseCase(
        loaders=loaders,
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
    embedder = build_embedder(config)
    store = build_vector_store(config)
    retriever = build_retriever(config, embedder, store)
    llm = build_llm(config)
    reranker = build_reranker(config)

    use_case = QueryGraphBuilder(retriever=retriever, llm=llm, reranker=reranker)
    result = use_case.execute(question)

    print(f"\nAnswer: {result.answer}\n")
    print("Sources:")
    for chunk in result.chunks:
        source = chunk.metadata.get("source", "unknown")
        print(f"  - {source}")
    if result.metadata.get("attempts", 0) > 0:
        print(f"\nRetrieval attempts: {result.metadata['attempts'] + 1}")

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
    config = AppConfig.from_yaml(args.config)

    if args.command == "ingest":
        cmd_ingest(config)
    elif args.command == "query":
        cmd_query(config, args.question)


if __name__ == "__main__":
    main()
