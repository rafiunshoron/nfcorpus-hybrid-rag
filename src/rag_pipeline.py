from dataclasses import dataclass
from time import perf_counter

from src.answer_generator import AnswerGenerator
from src.context_builder import build_context
from src.retriever import HybridRetriever


@dataclass(frozen=True, slots=True)
class SourceReference:
    number: int
    document_id: str
    title: str
    source_url: str | None
    cited: bool


@dataclass(frozen=True, slots=True)
class RAGResponse:
    answer: str
    sources: tuple[SourceReference, ...]
    insufficient_evidence: bool
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float


class MedicalRAGPipeline:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.answer_generator = AnswerGenerator()

    def ask(self, question: str) -> RAGResponse:
        question = question.strip()

        if not question:
            raise ValueError(
                "The question cannot be empty."
            )

        total_start = perf_counter()

        retrieval_start = perf_counter()

        retrieval_results = (
            self.retriever.retrieve(question)
        )

        context = build_context(retrieval_results)

        retrieval_seconds = (
            perf_counter() - retrieval_start
        )

        generation_start = perf_counter()

        generated_answer = (
            self.answer_generator.generate(
                question=question,
                context=context,
            )
        )

        generation_seconds = (
            perf_counter() - generation_start
        )

        cited_numbers = set(
            generated_answer.cited_source_numbers
        )

        sources = tuple(
            SourceReference(
                number=source_number,
                document_id=source.document_id,
                title=source.title,
                source_url=source.source_url,
                cited=source_number in cited_numbers,
            )
            for source_number, source in enumerate(
                context.sources,
                start=1,
            )
        )

        total_seconds = (
            perf_counter() - total_start
        )

        return RAGResponse(
            answer=generated_answer.text,
            sources=sources,
            insufficient_evidence=(
                generated_answer.insufficient_evidence
            ),
            retrieval_seconds=retrieval_seconds,
            generation_seconds=generation_seconds,
            total_seconds=total_seconds,
        )