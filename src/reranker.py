from dataclasses import dataclass

import numpy as np
from sentence_transformers import CrossEncoder

from src.config import (
    RERANK_TOP_K,
    RERANKER_BATCH_SIZE,
    RERANKER_DEVICE,
    RERANKER_MODEL_NAME,
)
from src.retriever import RetrievalResult


@dataclass(frozen=True, slots=True)
class RerankedResult:
    chunk: RetrievalResult
    rerank_score: float


class CrossEncoderReranker:
    def __init__(self) -> None:
        print(
            f"Loading reranker: {RERANKER_MODEL_NAME}"
        )

        self.model = CrossEncoder(
            RERANKER_MODEL_NAME,
            device=RERANKER_DEVICE,
        )

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankedResult]:
        query = query.strip()

        if not query:
            raise ValueError("The query cannot be empty.")

        if not candidates:
            return []

        query_chunk_pairs = [
            (query, candidate.content)
            for candidate in candidates
        ]

        scores = self.model.predict(
            query_chunk_pairs,
            batch_size=RERANKER_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        scores = np.asarray(scores).reshape(-1)

        if len(scores) != len(candidates):
            raise RuntimeError(
                "The reranker returned an unexpected "
                "number of scores."
            )

        reranked_results = [
            RerankedResult(
                chunk=candidate,
                rerank_score=float(score),
            )
            for candidate, score in zip(
                candidates,
                scores,
                strict=True,
            )
        ]

        reranked_results.sort(
            key=lambda result: result.rerank_score,
            reverse=True,
        )

        return reranked_results[:top_k]