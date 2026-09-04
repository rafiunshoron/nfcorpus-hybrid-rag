import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.config import CORPUS_PATH, QRELS_DIR, QUERIES_PATH


Corpus = dict[str, dict[str, Any]]
Queries = dict[str, dict[str, Any]]
Qrels = dict[str, dict[str, int]]


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {file_path} at line {line_number}."
                ) from error

            records.append(record)

    return records


def load_corpus() -> Corpus:
    corpus: Corpus = {}

    for record in load_jsonl(CORPUS_PATH):
        document_id = record.get("_id")
        title = record.get("title", "")
        text = record.get("text")
        metadata = record.get("metadata", {})

        if not document_id or not text:
            raise ValueError(
                "Every corpus record must contain '_id' and 'text'."
            )

        if document_id in corpus:
            raise ValueError(f"Duplicate document ID: {document_id}")

        if not isinstance(metadata, dict):
            raise ValueError(
                f"Metadata must be an object for document: {document_id}"
            )

        corpus[document_id] = {
            "title": title,
            "text": text,
            "metadata": metadata,
        }

    return corpus


def load_queries() -> Queries:
    queries: Queries = {}

    for record in load_jsonl(QUERIES_PATH):
        query_id = record.get("_id")
        text = record.get("text")
        metadata = record.get("metadata", {})

        if not query_id or not text:
            raise ValueError(
                "Every query record must contain '_id' and 'text'."
            )

        if query_id in queries:
            raise ValueError(f"Duplicate query ID: {query_id}")

        if not isinstance(metadata, dict):
            raise ValueError(
                f"Metadata must be an object for query: {query_id}"
            )

        queries[query_id] = {
            "text": text,
            "metadata": metadata,
        }

    return queries


def load_qrels(split: str) -> Qrels:
    qrels_path = QRELS_DIR / f"{split}.tsv"

    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found: {qrels_path}")

    qrels: Qrels = {}

    with qrels_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")

        expected_columns = {"query-id", "corpus-id", "score"}

        if set(reader.fieldnames or []) != expected_columns:
            raise ValueError(
                f"Unexpected qrels columns: {reader.fieldnames}"
            )

        for row in reader:
            query_id = row["query-id"]
            document_id = row["corpus-id"]
            score = int(row["score"])

            qrels.setdefault(query_id, {})[document_id] = score

    return qrels


def validate_dataset(
    corpus: Corpus,
    queries: Queries,
    qrels: Qrels,
) -> None:
    missing_queries = set(qrels) - set(queries)

    missing_documents = {
        document_id
        for relevant_documents in qrels.values()
        for document_id in relevant_documents
        if document_id not in corpus
    }

    if missing_queries:
        raise ValueError(
            f"Qrels contain {len(missing_queries)} unknown query IDs."
        )

    if missing_documents:
        raise ValueError(
            f"Qrels contain {len(missing_documents)} unknown document IDs."
        )


def load_nfcorpus(
    split: str,
) -> tuple[Corpus, Queries, Qrels]:
    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels(split)

    validate_dataset(corpus, queries, qrels)

    return corpus, queries, qrels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        default="dev",
    )
    args = parser.parse_args()

    corpus, queries, qrels = load_nfcorpus(args.split)

    relevance_judgments = sum(
        len(documents) for documents in qrels.values()
    )

    print(f"Corpus documents: {len(corpus)}")
    print(f"All queries: {len(queries)}")
    print(f"{args.split.title()} queries: {len(qrels)}")
    print(f"Relevance judgments: {relevance_judgments}")
    print("Dataset validation: passed")


if __name__ == "__main__":
    main()