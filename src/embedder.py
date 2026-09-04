import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
)


FloatArray = NDArray[np.float32]

QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)


class EmbeddingService:
    def __init__(self) -> None:
        print(
            f"Loading embedding model: {EMBEDDING_MODEL_NAME}"
        )

        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=EMBEDDING_DEVICE,
        )

        model_dimension = self.model.get_embedding_dimension()

        if model_dimension != EMBEDDING_DIMENSION:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {EMBEDDING_DIMENSION}, "
                f"but model produces {model_dimension}."
            )

    def embed_documents(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> FloatArray:
        if not texts:
            return np.empty(
                (0, EMBEDDING_DIMENSION),
                dtype=np.float32,
            )

        embeddings = self.model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def embed_query(self, query: str) -> FloatArray:
        query = query.strip()

        if not query:
            raise ValueError("The query cannot be empty.")

        instructed_query = f"{QUERY_INSTRUCTION}{query}"

        embedding = self.model.encode(
            instructed_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embedding,
            dtype=np.float32,
        )