import psycopg
from pgvector.psycopg import register_vector

from src.config import (
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)


def get_connection() -> psycopg.Connection:
    connection = psycopg.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
    )

    register_vector(connection)

    return connection


def verify_database_connection() -> None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT current_database(), extversion
                FROM pg_extension
                WHERE extname = 'vector';
                """
            )

            result = cursor.fetchone()

    if result is None:
        raise RuntimeError("The pgvector extension is not available.")

    database_name, vector_version = result

    print(f"Connected to database: {database_name}")
    print(f"pgvector version: {vector_version}")


if __name__ == "__main__":
    verify_database_connection()