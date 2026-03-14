import argparse
import sys

from rag_agent.config import AppConfig

from rag_agent.adapters.doc_loaders.markdown_loader import MarkdownLoader
from rag_agent.adapters.chunkers.fixed_size_chunker import FixedSizeChunker
from rag_agent.adapters.embedders.azure_openai_embedder import AzureOpenAIEmbedder
from rag_agent.adapters.vector_stores.chroma_store import ChromaStore
from rag_agent.adapters.retrievers.dense_retriever import DenseRetriever
from rag_agent.adapters.rerankers.no_reranker import NoReranker
from rag_agent.adapters.llms.azure_openai_llm import AzureOpenAILLM

from rag_agent.core.use_cases import IngestUseCase, QueryUseCase


def build_config(config_path: str) -> AppConfig:
    return AppConfig.from_yaml(config_path)


def cmd_ingest(config: AppConfig) -> None:
    loader = MarkdownLoader(path=config.data_source.path)
    chunker = FixedSizeChunker(
        chunk_size=config.chunker.chunk_size,
        chunk_overlap=config.chunker.chunk_overlap,
    )
    embedder = AzureOpenAIEmbedder(model=config.embedder.model)
    store = ChromaStore(
        collection_name=config.vector_store.collection_name,
        path=config.vector_store.path,
    )

    use_case = IngestUseCase(
        loader=loader, chunker=chunker, embedder=embedder, store=store
    )
    chunks = use_case.execute()
    print(f"Ingested {len(chunks)} chunks")


def cmd_query(config: AppConfig, question: str) -> None:
    embedder = AzureOpenAIEmbedder(model=config.embedder.model)
    store = ChromaStore(
        collection_name=config.vector_store.collection_name,
        path=config.vector_store.path,
    )
    retriever = DenseRetriever(
        embedder=embedder,
        vector_store=store,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Agent")
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to config YAML"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Ingest documents into vector store")

    query_parser = subparsers.add_parser("query", help="Query the RAG agent")
    query_parser.add_argument("question", help="The question to ask")

    args = parser.parse_args()
    config = build_config(args.config)

    if args.command == "ingest":
        cmd_ingest(config)
    elif args.command == "query":
        cmd_query(config, args.question)


if __name__ == "__main__":
    main()
