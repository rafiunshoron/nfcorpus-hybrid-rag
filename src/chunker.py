from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
)
from src.dataset_loader import Corpus


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    token_count: int


def create_text_splitter(
) -> tuple[PreTrainedTokenizerBase, RecursiveCharacterTextSplitter]:
    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL_NAME
    )

    text_splitter = (
        RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
            strip_whitespace=True,
        )
    )

    return tokenizer, text_splitter


def count_tokens(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
) -> int:
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def chunk_document(
    document_id: str,
    title: str,
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    text_splitter: RecursiveCharacterTextSplitter,
) -> list[DocumentChunk]:
    body_chunks = text_splitter.split_text(text)

    document_chunks = []

    for chunk_index, body in enumerate(body_chunks):
        if title:
            chunk_content = f"{title}\n\n{body}"
        else:
            chunk_content = body

        document_chunks.append(
            DocumentChunk(
                chunk_id=(
                    f"{document_id}-chunk-{chunk_index:04d}"
                ),
                document_id=document_id,
                chunk_index=chunk_index,
                content=chunk_content,
                token_count=count_tokens(
                    chunk_content,
                    tokenizer,
                ),
            )
        )

    return document_chunks


def chunk_corpus(corpus: Corpus) -> list[DocumentChunk]:
    tokenizer, text_splitter = create_text_splitter()

    all_chunks = []

    for document_id, document in corpus.items():
        document_chunks = chunk_document(
            document_id=document_id,
            title=document["title"],
            text=document["text"],
            tokenizer=tokenizer,
            text_splitter=text_splitter,
        )

        all_chunks.extend(document_chunks)

    return all_chunks