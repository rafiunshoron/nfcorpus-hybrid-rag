import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def require_environment_variable(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Required environment variable '{name}' is missing."
        )

    return value


# Database configuration
POSTGRES_DB = require_environment_variable("POSTGRES_DB")
POSTGRES_USER = require_environment_variable("POSTGRES_USER")
POSTGRES_PASSWORD = require_environment_variable(
    "POSTGRES_PASSWORD"
)
POSTGRES_HOST = require_environment_variable("POSTGRES_HOST")
POSTGRES_PORT = int(
    require_environment_variable("POSTGRES_PORT")
)


# Dataset paths
NFCORPUS_DIR = (
    PROJECT_ROOT / "data" / "raw" / "nfcorpus"
)
CORPUS_PATH = NFCORPUS_DIR / "corpus.jsonl"
QUERIES_PATH = NFCORPUS_DIR / "queries.jsonl"
QRELS_DIR = NFCORPUS_DIR / "qrels"


# Embedding configuration
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384
EMBEDDING_DEVICE = "cpu"
EMBEDDING_BATCH_SIZE = 32


# Chunking configuration
CHUNK_SIZE = 350
CHUNK_OVERLAP = 50


# Retrieval configuration
DENSE_TOP_K = 50
SPARSE_TOP_K = 50
HYBRID_TOP_K = 50
RRF_CONSTANT = 60
HNSW_EF_SEARCH = 100


# Reranking configuration
RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)
RERANKER_DEVICE = "cpu"
RERANKER_BATCH_SIZE = 16
RERANK_TOP_K = 50


# Answer-generation configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)