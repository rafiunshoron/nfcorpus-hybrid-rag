# NFCorpus Hybrid Medical RAG

An evidence-grounded biomedical question-answering system built
with PostgreSQL, pgvector, dense retrieval, full-text search,
Reciprocal Rank Fusion, and Groq-hosted GPT-OSS.

The project retrieves evidence from NFCorpus, generates concise
answers with inline citations, validates source labels, and refuses
questions when the retrieved evidence is insufficient.

> This project is intended for information-retrieval research and
> education. It does not provide medical advice.

## Key features

- Token-aware recursive document chunking
- CPU-compatible BGE embeddings
- PostgreSQL and pgvector storage
- HNSW approximate vector search
- PostgreSQL full-text sparse retrieval
- Equal-weight Reciprocal Rank Fusion
- Document-level result deduplication
- Structured Groq answer generation
- Inline citation validation
- Automatic citation-correction attempt
- Evidence-insufficient refusal behavior
- Dense, sparse, hybrid, and reranking evaluation
- Development/test split separation

## Architecture

```mermaid
flowchart TD
    Q["User question"] --> D["BGE query embedding"]
    Q --> S["PostgreSQL full-text search"]
    D --> H["pgvector HNSW search"]
    H --> F["Reciprocal Rank Fusion"]
    S --> F
    F --> C["Five unique documents"]
    C --> G["GPT-OSS 120B"]
    G --> V["Citation validation"]
    V --> A["Grounded answer"]