"""Agente RAG de FAQs para Parachute S.A.

Este programa NO carga el corpus. Depende de que el script de carga haya creado
la tabla `faqs` y almacenado embeddings de `all-MiniLM-L6-v2` (384 dimensiones).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer


SYSTEM_PROMPT = """Eres el asistente de atención al cliente de Parachute S.A.
Responde únicamente preguntas sobre el evento nacional de paracaidismo Guatemala
2026. Antes de contestar una pregunta sobre el evento, usa siempre la herramienta
`buscar_faqs`. Basa tu respuesta solamente en los resultados de la herramienta.
No inventes datos ni políticas: si los resultados no contienen la respuesta, dilo
con claridad y recomienda escribir a soporte@parachutesa.gt. Responde en español,
de forma breve y amable. Si mencionas información encontrada, cita el ID de la FAQ
entre paréntesis, por ejemplo: (FAQ-012)."""


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "buscar_faqs",
        "description": "Busca FAQs oficiales de Parachute S.A. relevantes a una consulta.",
        "parameters": {
            "type": "object",
            "properties": {
                "consulta": {
                    "type": "string",
                    "description": "Pregunta del usuario o una reformulación breve para buscarla.",
                },
                "limite": {
                    "type": "integer",
                    "description": "Cantidad de FAQs a devolver; entre 1 y 5.",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["consulta"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class Settings:
    database_url: str
    model: str
    api_key: str
    base_url: str | None
    embedding_model: str
    table_name: str

    @classmethod
    def from_env(cls) -> "Settings":
        # OPENAI_API_KEY sirve para OpenAI; NVIDIA_API_KEY mantiene compatibilidad
        # con el .env del proyecto anterior y con endpoints OpenAI-compatibles de NVIDIA.
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("NVIDIA_API_KEY")
        database_url = os.getenv("DATABASE_URL")
        model = os.getenv("MODEL")
        if not api_key or not database_url or not model:
            raise RuntimeError(
                "Faltan variables requeridas. Configure DATABASE_URL, MODEL y "
                "OPENAI_API_KEY (o NVIDIA_API_KEY) en .env."
            )
        table_name = os.getenv("FAQ_TABLE", "faqs")
        if not table_name.isidentifier():
            raise RuntimeError("FAQ_TABLE debe ser un identificador SQL simple, por ejemplo: faqs")
        return cls(
            database_url=database_url,
            model=model,
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            table_name=table_name,
        )


class FAQSearcher:
    """Implementación local de la herramienta que consulta PostgreSQL + pgvector."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.encoder = SentenceTransformer(settings.embedding_model)

    def search(self, consulta: str, limite: int = 3) -> list[dict[str, Any]]:
        limite = max(1, min(int(limite), 5))
        embedding = self.encoder.encode(consulta, normalize_embeddings=True).tolist()
        # table_name fue validado con isidentifier(); los demás valores se parametrizan.
        query = f"""
            SELECT id, categoria, pregunta, respuesta, metadata,
                   1 - (embedding <=> %s::vector) AS similitud
            FROM {self.settings.table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        with psycopg.connect(self.settings.database_url) as connection:
            register_vector(connection)
            with connection.cursor() as cursor:
                cursor.execute(query, (embedding, embedding, limite))
                columns = [column.name for column in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]


def serializar_resultados(resultados: list[dict[str, Any]]) -> str:
    if not resultados:
        return json.dumps({"resultados": [], "mensaje": "No se encontraron FAQs."}, ensure_ascii=False)
    return json.dumps({"resultados": resultados}, ensure_ascii=False, default=str)


def responder(client: OpenAI, searcher: FAQSearcher, messages: list[dict[str, Any]], model: str) -> str:
    """Ejecuta el ciclo de function calling hasta que el modelo entregue texto."""
    for _ in range(3):
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[SEARCH_TOOL],
            tool_choice="auto",
            temperature=0.2,
        )
        message = completion.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return message.content or "No pude generar una respuesta. Intenta nuevamente."

        for tool_call in message.tool_calls:
            if tool_call.function.name != "buscar_faqs":
                output = json.dumps({"error": "Herramienta no permitida."})
            else:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    output = serializar_resultados(
                        searcher.search(arguments["consulta"], arguments.get("limite", 3))
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    output = json.dumps({"error": f"Argumentos inválidos: {error}"}, ensure_ascii=False)
                except psycopg.Error:
                    output = json.dumps(
                        {"error": "No fue posible consultar la base de conocimiento."}, ensure_ascii=False
                    )
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})
    return "No pude completar la búsqueda. Por favor, intenta formular la pregunta de otra manera."


def main() -> None:
    load_dotenv()
    try:
        settings = Settings.from_env()
        client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        searcher = FAQSearcher(settings)
    except RuntimeError as error:
        sys.exit(f"Error de configuración: {error}")

    print("Asistente de Parachute S.A. — escriba 'salir' para terminar.")
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            question = input("\nTú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break
        if question.lower() in {"salir", "exit", "quit"}:
            print("Hasta luego.")
            break
        if not question:
            continue
        messages.append({"role": "user", "content": question})
        try:
            print(f"\nAsistente: {responder(client, searcher, messages, settings.model)}")
        except Exception as error:  # Evita cerrar el chat por un fallo transitorio del proveedor.
            print(f"\nAsistente: Ocurrió un error al procesar la consulta: {error}")


if __name__ == "__main__":
    main()
