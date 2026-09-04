from src.chunker import DocumentChunk, chunk_corpus
from src.database import get_connection
from src.dataset_loader import Corpus, load_corpus
from src.embedder import EmbeddingService


def create_document_rows(
    corpus: Corpus,
) -> list[tuple[str, str, str, str | None]]:
    rows = []

    for document_id, document in corpus.items():
        source_url = document["metadata"].get("url")

        rows.append(
            (
                document_id,
                document["title"],
                document["text"],
                source_url,
            )
        )

    return rows


def create_chunk_rows(
    chunks: list[DocumentChunk],
    embeddings,
) -> list[tuple]:
    if len(chunks) != len(embeddings):
        raise ValueError(
            "The number of chunks and embeddings does not match."
        )

    rows = []

    for chunk, embedding in zip(
        chunks,
        embeddings,
        strict=True,
    ):
        rows.append(
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.chunk_index,
                chunk.content,
                chunk.token_count,
                embedding,
            )
        )

    return rows


def rebuild_index() -> None:
    print("Loading NFCorpus...")
    corpus = load_corpus()

    print(f"Documents loaded: {len(corpus)}")

    print("Applying recursive chunking...")
    chunks = chunk_corpus(corpus)

    if not chunks:
        raise RuntimeError("Chunking produced no chunks.")

    maximum_token_count = max(
        chunk.token_count for chunk in chunks
    )

    print(f"Chunks created: {len(chunks)}")
    print(f"Maximum chunk token count: {maximum_token_count}")

    if maximum_token_count > 512:
        raise RuntimeError(
            "At least one chunk exceeds the model's "
            "512-token limit."
        )

    embedding_service = EmbeddingService()

    print("Generating document embeddings...")

    embeddings = embedding_service.embed_documents(
        [chunk.content for chunk in chunks]
    )

    document_rows = create_document_rows(corpus)
    chunk_rows = create_chunk_rows(chunks, embeddings)

    print("Saving documents and chunks to PostgreSQL...")

    with get_connection() as connection:
        with connection.cursor() as cursor:
            # This script performs a complete deterministic rebuild.
            cursor.execute(
                "TRUNCATE TABLE chunks, documents CASCADE;"
            )

            cursor.executemany(
                """
                INSERT INTO documents
                (
                    document_id,
                    title,
                    content,
                    source_url
                )
                VALUES (%s, %s, %s, %s);
                """,
                document_rows,
            )

            cursor.executemany(
                """
                INSERT INTO chunks
                (
                    chunk_id,
                    document_id,
                    chunk_index,
                    content,
                    token_count,
                    embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                chunk_rows,
            )

    print("Indexing completed successfully.")
    print(f"Stored documents: {len(document_rows)}")
    print(f"Stored chunks: {len(chunk_rows)}")


if __name__ == "__main__":
    rebuild_index()