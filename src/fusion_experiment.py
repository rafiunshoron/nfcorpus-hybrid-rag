import argparse

from src.config import RRF_CONSTANT
from src.dataset_loader import load_nfcorpus
from src.evaluator import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from src.retriever import HybridRetriever, RetrievalResult


EVALUATION_K = 10

WEIGHT_CONFIGURATIONS = [
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1),
]


def weighted_rrf_document_ranking(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    dense_weight: float,
    sparse_weight: float,
) -> list[str]:
    chunk_scores: dict[str, float] = {}
    chunk_results: dict[str, RetrievalResult] = {}

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):
        chunk_results[result.chunk_id] = result

        chunk_scores[result.chunk_id] = (
            chunk_scores.get(result.chunk_id, 0.0)
            + dense_weight / (RRF_CONSTANT + rank)
        )

    for rank, result in enumerate(
        sparse_results,
        start=1,
    ):
        chunk_results.setdefault(
            result.chunk_id,
            result,
        )

        chunk_scores[result.chunk_id] = (
            chunk_scores.get(result.chunk_id, 0.0)
            + sparse_weight / (RRF_CONSTANT + rank)
        )

    ranked_chunk_ids = sorted(
        chunk_scores,
        key=chunk_scores.get,
        reverse=True,
    )

    document_ranking = []
    seen_document_ids = set()

    for chunk_id in ranked_chunk_ids:
        document_id = chunk_results[
            chunk_id
        ].document_id

        if document_id in seen_document_ids:
            continue

        seen_document_ids.add(document_id)
        document_ranking.append(document_id)

    return document_ranking


def create_metric_store(
) -> dict[str, dict[str, float]]:
    return {
        f"{dense_weight:.1f}/{sparse_weight:.1f}": {
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
        }
        for dense_weight, sparse_weight
        in WEIGHT_CONFIGURATIONS
    }


def run_experiment(
    split: str,
    limit: int | None,
) -> None:
    _, queries, qrels = load_nfcorpus(split)

    query_ids = sorted(qrels)

    if limit is not None:
        query_ids = query_ids[:limit]

    if not query_ids:
        raise RuntimeError("No evaluation queries found.")

    retriever = HybridRetriever()
    metric_store = create_metric_store()

    for query_number, query_id in enumerate(
        query_ids,
        start=1,
    ):
        query = queries[query_id]["text"]
        relevance = qrels[query_id]

        dense_results = retriever.dense_search(query)
        sparse_results = retriever.sparse_search(query)

        for dense_weight, sparse_weight in (
            WEIGHT_CONFIGURATIONS
        ):
            label = (
                f"{dense_weight:.1f}/"
                f"{sparse_weight:.1f}"
            )

            ranking = weighted_rrf_document_ranking(
                dense_results=dense_results,
                sparse_results=sparse_results,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )

            metric_store[label]["precision"] += (
                precision_at_k(
                    ranking,
                    relevance,
                    EVALUATION_K,
                )
            )
            metric_store[label]["recall"] += (
                recall_at_k(
                    ranking,
                    relevance,
                    EVALUATION_K,
                )
            )
            metric_store[label]["mrr"] += (
                reciprocal_rank_at_k(
                    ranking,
                    relevance,
                    EVALUATION_K,
                )
            )
            metric_store[label]["ndcg"] += (
                ndcg_at_k(
                    ranking,
                    relevance,
                    EVALUATION_K,
                )
            )

        if (
            query_number == 1
            or query_number % 25 == 0
            or query_number == len(query_ids)
        ):
            print(
                f"Processed {query_number}/"
                f"{len(query_ids)} queries"
            )

    query_count = len(query_ids)

    print()
    print(
        f"Weighted RRF experiment: "
        f"{split} ({query_count} queries)"
    )
    print(
        "Weights shown as Dense/Sparse"
    )
    print()

    header = (
        f"{'Weights':<12}"
        f"{'P@10':>10}"
        f"{'Recall@10':>12}"
        f"{'MRR@10':>10}"
        f"{'NDCG@10':>12}"
    )

    print(header)
    print("-" * len(header))

    for label, metrics in metric_store.items():
        print(
            f"{label:<12}"
            f"{metrics['precision'] / query_count:>10.4f}"
            f"{metrics['recall'] / query_count:>12.4f}"
            f"{metrics['mrr'] / query_count:>10.4f}"
            f"{metrics['ndcg'] / query_count:>12.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        default="dev",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError(
            "--limit must be greater than zero."
        )

    run_experiment(
        split=args.split,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()