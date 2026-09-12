"""Carga las FAQs de Parachute S.A. en PostgreSQL con embeddings vectoriales."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.types.json import Jsonb
from sentence_transformers import SentenceTransformer


EXPECTED_EMBEDDING_DIMENSION = 384
RECORD_SEPARATOR = re.compile(r"^-{20,}\s*$", re.MULTILINE)
RECORD_PATTERN = re.compile(
    r"""ID:\s*(?P<id>[^\n]+)\n
        CATEGORÍA:\s*(?P<categoria>[^\n]+)\n
        PREGUNTA:\s*(?P<pregunta>[^\n]+)\n
        RESPUESTA:\s*(?P<respuesta>.*?)\n
        METADATA:\s*(?P<metadata>\{.*\})\s*\Z""",
    re.DOTALL | re.VERBOSE,
)


@dataclass(frozen=True)
class FAQ:
    id: str
    categoria: str
    pregunta: str
    respuesta: str
    metadata: dict[str, Any]

    def text_for_embedding(self) -> str:
        return (
            f"Categoría: {self.categoria}\n"
            f"Pregunta: {self.pregunta}\n"
            f"Respuesta: {self.respuesta}"
        )


@dataclass(frozen=True)
class Settings:
    database_url: str
    embedding_model: str
    table_name: str
    schema_path: Path

    @classmethod
    def from_env(cls) -> "Settings":
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError("Falta DATABASE_URL en .env.")
        table_name = os.getenv("FAQ_TABLE", "faqs")
        if not table_name.isidentifier():
            raise RuntimeError("FAQ_TABLE debe ser un identificador SQL simple.")
        return cls(
            database_url=database_url,
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            table_name=table_name,
            schema_path=Path(__file__).resolve().parent / "db" / "schema.sql",
        )


def parse_faqs(corpus_path: Path) -> list[FAQ]:
    if not corpus_path.is_file():
        raise RuntimeError(f"No se encontró el corpus: {corpus_path}")

    corpus = corpus_path.read_text(encoding="utf-8")
    first_record = re.search(r"^ID:\s*", corpus, re.MULTILINE)
    if not first_record:
        raise RuntimeError("El corpus no contiene registros que comiencen con 'ID:'.")
    records = RECORD_SEPARATOR.split(corpus[first_record.start() :])
    faqs: list[FAQ] = []
    errors: list[str] = []

    for position, record in enumerate(records, start=1):
        record = record.strip()
        if not record or not record.startswith("ID:"):
            continue
        match = RECORD_PATTERN.fullmatch(record)
        if not match:
            errors.append(f"Registro {position}: formato no reconocido.")
            continue
        values = {name: value.strip() for name, value in match.groupdict().items() if name != "metadata"}
        if any(not value for value in values.values()):
            errors.append(f"Registro {position}: hay un campo de texto vacío.")
            continue
        try:
            metadata = json.loads(match.group("metadata"))
        except json.JSONDecodeError as error:
            errors.append(f"Registro {position}: metadata no es JSON válido ({error.msg}).")
            continue
        if not isinstance(metadata, dict):
            errors.append(f"Registro {position}: metadata debe ser un objeto JSON.")
            continue
        faqs.append(FAQ(metadata=metadata, **values))

    if errors:
        raise RuntimeError("El corpus contiene errores:\n- " + "\n- ".join(errors))
    if not faqs:
        raise RuntimeError("No se encontraron FAQs en el corpus.")
    if len({faq.id for faq in faqs}) != len(faqs):
        raise RuntimeError("El corpus contiene IDs de FAQ duplicados.")
    return faqs


def load_embeddings(model_name: str, faqs: list[FAQ], batch_size: int) -> list[list[float]]:
    model = SentenceTransformer(model_name)
    dimension = model.get_sentence_embedding_dimension()
    if dimension != EXPECTED_EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"El modelo {model_name} genera vectores de {dimension} dimensiones; "
            f"la tabla requiere {EXPECTED_EMBEDDING_DIMENSION}."
        )
    embeddings = model.encode(
        [faq.text_for_embedding() for faq in faqs],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.tolist()


def apply_schema(connection: psycopg.Connection[Any], schema_path: Path) -> None:
    if not schema_path.is_file():
        raise RuntimeError(f"No se encontró el esquema SQL: {schema_path}")
    connection.execute(schema_path.read_text(encoding="utf-8"))


def save_faqs(
    connection: psycopg.Connection[Any], table_name: str, faqs: list[FAQ], embeddings: list[list[float]]
) -> None:
    if len(faqs) != len(embeddings):
        raise RuntimeError("La cantidad de embeddings no coincide con la cantidad de FAQs.")
    query = sql.SQL(
        """
        INSERT INTO {} (id, categoria, pregunta, respuesta, metadata, embedding)
        VALUES (%s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (id) DO UPDATE SET
            categoria = EXCLUDED.categoria,
            pregunta = EXCLUDED.pregunta,
            respuesta = EXCLUDED.respuesta,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding
        """
    ).format(sql.Identifier(table_name))
    rows = [
        (faq.id, faq.categoria, faq.pregunta, faq.respuesta, Jsonb(faq.metadata), embedding)
        for faq, embedding in zip(faqs, embeddings, strict=True)
    ]
    with connection.cursor() as cursor:
        cursor.executemany(query, rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Carga FAQs y sus embeddings en PostgreSQL.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "Corpus_FAQs_Parachute_SA_2026.txt",
        help="Ruta al corpus de FAQs.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="FAQs procesadas por lote de embeddings.")
    arguments = parser.parse_args()
    if arguments.batch_size < 1:
        parser.error("--batch-size debe ser mayor que cero.")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
    try:
        settings = Settings.from_env()
        faqs = parse_faqs(arguments.corpus)
        embeddings = load_embeddings(settings.embedding_model, faqs, arguments.batch_size)
        with psycopg.connect(settings.database_url) as connection:
            apply_schema(connection, settings.schema_path)
            register_vector(connection)
            save_faqs(connection, settings.table_name, faqs, embeddings)
            connection.commit()
        print(f"Se cargaron {len(faqs)} FAQs en la tabla '{settings.table_name}'.")
    except (OSError, psycopg.Error, RuntimeError) as error:
        sys.exit(f"Error de carga: {error}")


if __name__ == "__main__":
    main()
