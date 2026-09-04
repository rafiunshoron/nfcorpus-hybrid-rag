from groq import APIError
from psycopg import Error as DatabaseError

from src.rag_pipeline import (
    MedicalRAGPipeline,
    RAGResponse,
)


EXIT_COMMANDS = {
    "exit",
    "quit",
    "q",
}


def display_response(
    response: RAGResponse,
) -> None:
    print()
    print("Answer")
    print("-" * 70)
    print(response.answer)

    cited_sources = [
        source
        for source in response.sources
        if source.cited
    ]

    if cited_sources:
        print()
        print("Cited sources")
        print("-" * 70)

        for source in cited_sources:
            print(
                f"[S{source.number}] "
                f"{source.document_id} - "
                f"{source.title}"
            )

            if source.source_url:
                print(
                    f"     {source.source_url}"
                )

    print()
    print("Performance")
    print("-" * 70)
    print(
        f"Retrieval:  "
        f"{response.retrieval_seconds:.3f} seconds"
    )
    print(
        f"Generation: "
        f"{response.generation_seconds:.3f} seconds"
    )
    print(
        f"Total:      "
        f"{response.total_seconds:.3f} seconds"
    )
    print()


def main() -> None:
    print("=" * 70)
    print("NFCorpus Hybrid Medical RAG")
    print("=" * 70)
    print(
        "Evidence-grounded biomedical question answering "
        "with cited sources."
    )
    print(
        "This application is for research and educational "
        "purposes, not medical advice."
    )
    print(
        "Type 'exit' or 'quit' to close the application."
    )
    print()

    try:
        pipeline = MedicalRAGPipeline()
    except (
        RuntimeError,
        DatabaseError,
        APIError,
    ) as error:
        print(
            f"Application startup failed: {error}"
        )
        return

    print()
    print("System ready.")

    while True:
        print()
        question = input("Question: ").strip()

        if question.lower() in EXIT_COMMANDS:
            print("Application closed.")
            break

        if not question:
            print(
                "Please enter a question."
            )
            continue

        try:
            response = pipeline.ask(question)
            display_response(response)

        except KeyboardInterrupt:
            print()
            print("Current question cancelled.")

        except DatabaseError as error:
            print(
                f"Database error: {error}"
            )

        except APIError as error:
            print(
                f"Groq API error: {error}"
            )

        except (
            RuntimeError,
            ValueError,
        ) as error:
            print(
                f"Processing error: {error}"
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Application closed.")