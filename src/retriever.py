from dataclasses import dataclass

from src.config import (
    DENSE_TOP_K,
    HNSW_EF_SEARCH,
    HYBRID_TOP_K,
    RRF_CONSTANT,
    SPARSE_TOP_K,
)
from src.database import get_connection
from src.embedder import EmbeddingService


@dataclass(slots=True)
class RetrievalResult:
    chunk_id: str
    document_id: str
    title: str
    source_url: str | None
    content: str
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float = 0.0


class HybridRetriever:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.embedding_service = (
            embedding_service or EmbeddingService()
        )

    def dense_search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
    ) -> list[RetrievalResult]:
        query = query.strip()

        if not query:
            raise ValueError(
                "The query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_embedding = (
            self.embedding_service.embed_query(query)
        )

        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT set_config(
                        'hnsw.ef_search',
                        %s,
                        true
                    );
                    """,
                    (str(HNSW_EF_SEARCH),),
                )

                cursor.execute(
                    """
                    SELECT
                        c.chunk_id,
                        c.document_id,
                        d.title,
                        d.source_url,
                        c.content,
                        c.embedding <=> %s AS distance
                    FROM chunks AS c
                    JOIN documents AS d
                        ON d.document_id = c.document_id
                    ORDER BY c.embedding <=> %s
                    LIMIT %s;
                    """,
                    (
                        query_embedding,
                        query_embedding,
                        top_k,
                    ),
                )

                rows = cursor.fetchall()

        return [
            RetrievalResult(
                chunk_id=row[0],
                document_id=row[1],
                title=row[2],
                source_url=row[3],
                content=row[4],
                dense_score=1.0 - float(row[5]),
            )
            for row in rows
        ]

    def sparse_search(
        self,
        query: str,
        top_k: int = SPARSE_TOP_K,
    ) -> list[RetrievalResult]:
        query = query.strip()

        if not query:
            raise ValueError(
                "The query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    WITH search_query AS
                    (
                        SELECT websearch_to_tsquery(
                            'english',
                            %s
                        ) AS query
                    )
                    SELECT
                        c.chunk_id,
                        c.document_id,
                        d.title,
                        d.source_url,
                        c.content,
                        ts_rank_cd(
                            c.search_vector,
                            search_query.query
                        ) AS score
                    FROM chunks AS c
                    JOIN documents AS d
                        ON d.document_id = c.document_id
                    CROSS JOIN search_query
                    WHERE
                        c.search_vector
                        @@ search_query.query
                    ORDER BY score DESC
                    LIMIT %s;
                    """,
                    (query, top_k),
                )

                rows = cursor.fetchall()

        return [
            RetrievalResult(
                chunk_id=row[0],
                document_id=row[1],
                title=row[2],
                source_url=row[3],
                content=row[4],
                sparse_score=float(row[5]),
            )
            for row in rows
        ]

    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int = HYBRID_TOP_K,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        combined_results: dict[
            str,
            RetrievalResult,
        ] = {}

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            combined_results[result.chunk_id] = result

            result.fusion_score += (
                1.0 / (RRF_CONSTANT + rank)
            )

        for rank, result in enumerate(
            sparse_results,
            start=1,
        ):
            if result.chunk_id in combined_results:
                combined_result = combined_results[
                    result.chunk_id
                ]

                combined_result.sparse_score = (
                    result.sparse_score
                )
            else:
                combined_results[result.chunk_id] = (
                    result
                )
                combined_result = result

            combined_result.fusion_score += (
                1.0 / (RRF_CONSTANT + rank)
            )

        ranked_results = sorted(
            combined_results.values(),
            key=lambda result: result.fusion_score,
            reverse=True,
        )

        return ranked_results[:top_k]

    def retrieve(
        self,
        query: str,
    ) -> list[RetrievalResult]:
        query = query.strip()

        if not query:
            raise ValueError(
                "The query cannot be empty."
            )

        dense_results = self.dense_search(query)
        sparse_results = self.sparse_search(query)

        return self.reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
        )