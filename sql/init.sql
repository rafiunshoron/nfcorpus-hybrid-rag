CREATE EXTENSION IF NOT EXISTS vector;


CREATE TABLE IF NOT EXISTS documents
(
    document_id TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    source_url  TEXT
);


CREATE TABLE IF NOT EXISTS chunks
(
    chunk_id      TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL
                  REFERENCES documents(document_id)
                  ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL CHECK (chunk_index >= 0),
    content       TEXT NOT NULL,
    token_count   INTEGER NOT NULL CHECK (token_count > 0),
    embedding     VECTOR(384) NOT NULL,

    search_vector TSVECTOR GENERATED ALWAYS AS
    (
        to_tsvector('english', content)
    ) STORED,

    UNIQUE (document_id, chunk_index)
);


CREATE INDEX IF NOT EXISTS chunks_document_id_index
ON chunks (document_id);


CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_index
ON chunks
USING hnsw (embedding vector_cosine_ops);


CREATE INDEX IF NOT EXISTS chunks_search_vector_gin_index
ON chunks
USING gin (search_vector);