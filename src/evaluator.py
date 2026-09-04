import argparse
import math
from collections.abc import Sequence
from time import perf_counter

from src.dataset_loader import Qrels, load_nfcorpus
from src.reranker import CrossEncoderReranker, RerankedResult
from src.retriever import HybridRetriever, RetrievalResult


EVALUATION_K = 10


def unique_document_ids(
    results: Sequence[RetrievalResult],
) -> list[str]:
    document_ids = []
    seen_ids = set()

    for result in results:
        if result.document_id in seen_ids:
            continue

        seen_ids.add(result.document_id)
        document_ids.append(result.document_id)

    return document_ids


def unique_reranked_document_ids(
    results: Sequence[RerankedResult],
) -> list[str]:
    document_ids = []
    seen_ids = set()

    for result in results:
        document_id = result.chunk.document_id

        if document_id in seen_ids:
            continue

        seen_ids.add(document_id)
        document_ids.append(document_id)

    return document_ids


def precision_at_k(
    ranking: list[str],
    relevance: dict[str, int],
    k: int,
) -> float:
    relevant_documents = {
        document_id
        for document_id, score in relevance.items()
        if score > 0
    }

    retrieved = ranking[:k]

    relevant_retrieved = sum(
        document_id in relevant_documents
        for document_id in retrieved
    )

    return relevant_retrieved / k


def recall_at_k(
    ranking: list[str],
    relevance: dict[str, int],
    k: int,
) -> float:
    relevant_documents = {
        document_id
        for document_id, score in relevance.items()
        if score > 0
    }

    if not relevant_documents:
        return 0.0

    relevant_retrieved = sum(
        document_id in relevant_documents
        for document_id in ranking[:k]
    )

    return relevant_retrieved / len(relevant_documents)


def reciprocal_rank_at_k(
    ranking: list[str],
    relevance: dict[str, int],
    k: int,
) -> float:
    for rank, document_id in enumerate(
        ranking[:k],
        start=1,
    ):
        if relevance.get(document_id, 0) > 0:
            return 1.0 / rank

    return 0.0


def dcg_at_k(
    ranking: list[str],
    relevance: dict[str, int],
    k: int,
) -> float:
    score = 0.0

    for rank, document_id in enumerate(
        ranking[:k],
        start=1,
    ):
        relevance_score = relevance.get(document_id, 0)

        score += (
            (2**relevance_score - 1)
            / math.log2(rank + 1)
        )

    return score


def ndcg_at_k(
    ranking: list[str],
    relevance: dict[str, int],
    k: int,
) -> float:
    actual_dcg = dcg_at_k(
        ranking=ranking,
        relevance=relevance,
        k=k,
    )

    ideal_scores = sorted(
        relevance.values(),
        reverse=True,
    )[:k]

    ideal_dcg = sum(
        (2**score - 1) / math.log2(rank + 1)
        for rank, score in enumerate(
            ideal_scores,
            start=1,
        )
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def create_metric_store() -> dict[str, dict[str, float]]:
    stages = [
        "Dense",
        "Sparse",
        "Hybrid",
        "Reranked",
    ]

    return {
        stage: {
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "latency": 0.0,
        }
        for stage in stages
    }


def update_metrics(
    metric_store: dict[str, dict[str, float]],
    stage: str,
    ranking: list[str],
    relevance: dict[str, int],
    latency: float,
) -> None:
    metrics = metric_store[stage]

    metrics["precision"] += precision_at_k(
        ranking,
        relevance,
        EVALUATION_K,
    )
    metrics["recall"] += recall_at_k(
        ranking,
        relevance,
        EVALUATION_K,
    )
    metrics["mrr"] += reciprocal_rank_at_k(
        ranking,
        relevance,
        EVALUATION_K,
    )
    metrics["ndcg"] += ndcg_at_k(
        ranking,
        relevance,
        EVALUATION_K,
    )
    metrics["latency"] += latency


def evaluate(
    split: str,
    limit: int | None,
) -> None:
    _, queries, qrels = load_nfcorpus(split)

    query_ids = sorted(qrels)

    if limit is not None:
        query_ids = query_ids[:limit]

    if not query_ids:
        raise RuntimeError("No evaluation queries were found.")

    retriever = HybridRetriever()
    reranker = CrossEncoderReranker()

    metrics = create_metric_store()

    for query_number, query_id in enumerate(
        query_ids,
        start=1,
    ):
        query = queries[query_id]["text"]
        relevance = qrels[query_id]

        print(
            f"[{query_number}/{len(query_ids)}] "
            f"Evaluating {query_id}"
        )

        start = perf_counter()
        dense_results = retriever.dense_search(query)
        dense_latency = perf_counter() - start

        start = perf_counter()
        sparse_results = retriever.sparse_search(query)
        sparse_latency = perf_counter() - start

        start = perf_counter()
        hybrid_results = (
            retriever.reciprocal_rank_fusion(
                dense_results=dense_results,
                sparse_results=sparse_results,
            )
        )
        fusion_latency = perf_counter() - start

        start = perf_counter()
        reranked_results = reranker.rerank(
            query=query,
            candidates=hybrid_results,
        )
        reranking_latency = perf_counter() - start

        dense_ranking = unique_document_ids(
            dense_results
        )
        sparse_ranking = unique_document_ids(
            sparse_results
        )
        hybrid_ranking = unique_document_ids(
            hybrid_results
        )
        reranked_ranking = unique_reranked_document_ids(
            reranked_results
        )

        hybrid_latency = (
            dense_latency
            + sparse_latency
            + fusion_latency
        )

        update_metrics(
            metrics,
            "Dense",
            dense_ranking,
            relevance,
            dense_latency,
        )
        update_metrics(
            metrics,
            "Sparse",
            sparse_ranking,
            relevance,
            sparse_latency,
        )
        update_metrics(
            metrics,
            "Hybrid",
            hybrid_ranking,
            relevance,
            hybrid_latency,
        )
        update_metrics(
            metrics,
            "Reranked",
            reranked_ranking,
            relevance,
            hybrid_latency + reranking_latency,
        )

    query_count = len(query_ids)

    print()
    print(
        f"Evaluation split: {split} "
        f"({query_count} queries)"
    )
    print(f"Metrics calculated at K={EVALUATION_K}")
    print()

    header = (
        f"{'Stage':<12}"
        f"{'P@10':>10}"
        f"{'Recall@10':>12}"
        f"{'MRR@10':>10}"
        f"{'NDCG@10':>12}"
        f"{'Avg sec':>10}"
    )

    print(header)
    print("-" * len(header))

    for stage, stage_metrics in metrics.items():
        print(
            f"{stage:<12}"
            f"{stage_metrics['precision'] / query_count:>10.4f}"
            f"{stage_metrics['recall'] / query_count:>12.4f}"
            f"{stage_metrics['mrr'] / query_count:>10.4f}"
            f"{stage_metrics['ndcg'] / query_count:>12.4f}"
            f"{stage_metrics['latency'] / query_count:>10.3f}"
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
        help="Optional number of queries for a quick test.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero.")

    evaluate(
        split=args.split,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()