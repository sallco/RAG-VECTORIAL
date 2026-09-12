"""Agente RAG de FAQs para Parachute S.A.

Este programa NO carga el corpus. Depende de que el script de carga haya creado
la tabla `faqs` y almacenado embeddings de
`paraphrase-multilingual-MiniLM-L12-v2` (384 dimensiones).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer


SYSTEM_PROMPT = """Eres el asistente de atención al cliente de Parachute S.A.
Responde únicamente preguntas sobre el evento nacional de paracaidismo Guatemala
2026. Antes de contestar una pregunta sobre el evento, usa siempre la herramienta
`buscar_faqs`. Basa tu respuesta solamente en los resultados de la herramienta.
No inventes datos ni políticas: si la herramienta no devuelve resultados, responde
que no puedes contestar esa pregunta con la información disponible y recomienda
escribir a soporte@parachutesa.gt. Responde en español, de forma breve y amable.
Si mencionas información encontrada, cita el ID de la FAQ entre paréntesis, por
ejemplo: (FAQ-012)."""


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
    minimum_similarity: float

    @classmethod
    def from_env(cls) -> "Settings":
        # OPENAI_API_KEY sirve para OpenAI; NVIDIA_API_KEY mantiene compatibilidad
        # con el .env del proyecto anterior y con endpoints OpenAI-compatibles de NVIDIA.
        base_url = os.getenv("OPENAI_BASE_URL") or None
        if base_url and "nvidia.com" in base_url.lower():
            api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
        else:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("NVIDIA_API_KEY")
        database_url = os.getenv("DATABASE_URL")
        model = os.getenv("MODEL")
        embedding_model = os.getenv("EMBEDDING_MODEL")
        minimum_similarity_raw = os.getenv("MINIMUM_SIMILARITY")
        missing: list[str] = []
        if not database_url:
            missing.append("DATABASE_URL")
        if not model:
            missing.append("MODEL")
        if not api_key:
            missing.append("NVIDIA_API_KEY u OPENAI_API_KEY")
        if not embedding_model:
            missing.append("EMBEDDING_MODEL")
        if not minimum_similarity_raw:
            missing.append("MINIMUM_SIMILARITY")
        table_name = os.getenv("FAQ_TABLE")
        if not table_name:
            missing.append("FAQ_TABLE")
        if missing:
            raise RuntimeError(f"Faltan en .env: {', '.join(missing)}.")
        if not table_name.isidentifier():
            raise RuntimeError("FAQ_TABLE debe ser un identificador SQL simple, por ejemplo: faqs")
        try:
            minimum_similarity = float(minimum_similarity_raw)
        except ValueError as error:
            raise RuntimeError("MINIMUM_SIMILARITY debe ser un número entre 0 y 1.") from error
        if not 0 <= minimum_similarity <= 1:
            raise RuntimeError("MINIMUM_SIMILARITY debe estar entre 0 y 1.")
        return cls(
            database_url=database_url,
            model=model,
            api_key=api_key,
            base_url=base_url,
            embedding_model=embedding_model,
            table_name=table_name,
            minimum_similarity=minimum_similarity,
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
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                for result in results:
                    result["similitud"] = float(result["similitud"])
                return [
                    result
                    for result in results
                    if result["similitud"] >= self.settings.minimum_similarity
                ]


def serializar_resultados(resultados: list[dict[str, Any]]) -> str:
    if not resultados:
        return json.dumps(
            {
                "resultados": [],
                "mensaje": "No se encontraron FAQs con suficiente similitud para responder.",
            },
            ensure_ascii=False,
        )
    return json.dumps({"resultados": resultados}, ensure_ascii=False, default=str)


def responder(client: OpenAI, searcher: FAQSearcher, messages: list[dict[str, Any]], model: str) -> str:
    """Ejecuta el ciclo de function calling hasta que el modelo entregue texto."""
    tool_used = False
    for _ in range(3):
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[SEARCH_TOOL],
            # Después de consultar PostgreSQL se fuerza la respuesta final. Esto
            # evita que algunos modelos compatibles vuelvan a llamar la función.
            tool_choice=(
                "none"
                if tool_used
                else {"type": "function", "function": {"name": "buscar_faqs"}}
            ),
            temperature=0.2,
        )
        message = completion.choices[0].message
        # No reenviar el model_dump completo: proveedores OpenAI-compatibles
        # pueden rechazar con HTTP 400 campos adicionales de versiones nuevas
        # del SDK. Solo se conserva el formato estándar requerido.
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": message.content,
        }
        if message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        messages.append(assistant_message)

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
        tool_used = True
    return "No pude completar la búsqueda. Por favor, intenta formular la pregunta de otra manera."


def main() -> None:
    # La configuración del proyecto debe prevalecer sobre variables antiguas
    # definidas en Windows o heredadas de otros ejercicios. La ruta se resuelve
    # junto a este script para permitir ejecutarlo desde cualquier directorio.
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        sys.exit(f"Error de configuración: no se encontró {env_path}")
    load_dotenv(env_path, override=True)
    try:
        settings = Settings.from_env()
        client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=120.0,
            max_retries=1,
        )
        searcher = FAQSearcher(settings)
    except RuntimeError as error:
        sys.exit(f"Error de configuración: {error}")

    print("Asistente de Parachute S.A. — escriba 'Bye' para terminar.")
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            question = input("\nTú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break
        if question.lower() == "bye":
            print("Hasta luego.")
            break
        if not question:
            continue
        messages.append({"role": "user", "content": question})
        try:
            print(f"\nAsistente: {responder(client, searcher, messages, settings.model)}")
        except APITimeoutError:
            print("\nAsistente: El proveedor tardó demasiado en responder. Intenta nuevamente.")
        except APIConnectionError:
            print("\nAsistente: No fue posible conectarse con el proveedor del modelo.")
        except APIStatusError as error:
            detail = ""
            if isinstance(error.body, dict):
                provider_error = error.body.get("error", error.body)
                if isinstance(provider_error, dict):
                    detail = str(provider_error.get("message", ""))
                elif provider_error:
                    detail = str(provider_error)
            suffix = f" Detalle: {detail}" if detail else ""
            print(
                f"\nAsistente: El proveedor respondió con error HTTP "
                f"{error.status_code}.{suffix}"
            )
        except Exception as error:  # Evita cerrar el chat por un fallo no previsto.
            print(f"\nAsistente: Ocurrió un error al procesar la consulta: {error}")


if __name__ == "__main__":
    main()
