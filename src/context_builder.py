from dataclasses import dataclass

from src.retriever import RetrievalResult


CONTEXT_TOP_K = 5


@dataclass(frozen=True, slots=True)
class ContextBundle:
    text: str
    sources: tuple[RetrievalResult, ...]


def select_unique_documents(
    results: list[RetrievalResult],
    top_k: int = CONTEXT_TOP_K,
) -> list[RetrievalResult]:
    selected_results = []
    seen_document_ids = set()

    for result in results:
        if result.document_id in seen_document_ids:
            continue

        seen_document_ids.add(result.document_id)
        selected_results.append(result)

        if len(selected_results) == top_k:
            break

    return selected_results


def build_context(
    results: list[RetrievalResult],
    top_k: int = CONTEXT_TOP_K,
) -> ContextBundle:
    selected_results = select_unique_documents(
        results=results,
        top_k=top_k,
    )

    context_sections = []

    for source_number, result in enumerate(
        selected_results,
        start=1,
    ):
        source_parts = [
            f"[S{source_number}]",
            f"Document ID: {result.document_id}",
            f"Title: {result.title}",
        ]

        if result.source_url:
            source_parts.append(
                f"Source URL: {result.source_url}"
            )

        source_parts.append(
            f"Content: {result.content}"
        )

        context_sections.append(
            "\n".join(source_parts)
        )

    return ContextBundle(
        text="\n\n".join(context_sections),
        sources=tuple(selected_results),
    )