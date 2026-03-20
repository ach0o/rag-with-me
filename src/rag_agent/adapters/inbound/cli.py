import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Agent")
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to config YAML"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Ingest documents into vector store")

    query_parser = subparsers.add_parser("query", help="Query the RAG agent")
    query_parser.add_argument("question", help="The question to ask")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the RAG agent")
    eval_parser.add_argument(
        "--dataset", default="eval_data/gold_qa.json", help="Path to dataset JSON"
    )

    args = parser.parse_args()
    return args
