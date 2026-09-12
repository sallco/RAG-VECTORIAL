"""Agente RAG de FAQs para Parachute S.A.

Este programa NO carga el corpus. Depende de que el script de carga haya creado
la tabla `faqs` y almacenado embeddings de
`paraphrase-multilingual-MiniLM-L12-v2` (384 dimensiones).
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
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
        consulta_normalizada = re.sub(r"([?!])\1+", r"\1", consulta.strip())
        embedding = self.encoder.encode(consulta_normalizada, normalize_embeddings=True).tolist()
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


def ultima_pregunta_usuario(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message["role"] == "user" and isinstance(message.get("content"), str):
            question = message["content"].strip()
            if question:
                return question
    raise ValueError("No se encontró una pregunta del usuario para consultar.")


def buscar_combinado(
    searcher: FAQSearcher, consulta_reformulada: str, pregunta_original: str, limite: int
) -> list[dict[str, Any]]:
    """Busca con la reformulación del LLM (más limpia para frases ruidosas) y usa
    la pregunta original del usuario como respaldo si esa búsqueda no encuentra nada."""
    resultados = searcher.search(consulta_reformulada, limite)
    if resultados:
        return resultados
    if pregunta_original.strip().lower() == consulta_reformulada.strip().lower():
        return resultados
    return searcher.search(pregunta_original, limite)


def parsear_argumentos_busqueda(raw_arguments: str) -> tuple[str, int]:
    arguments = json.loads(raw_arguments)
    if not isinstance(arguments, dict):
        raise TypeError("Los argumentos de la herramienta deben ser un objeto JSON.")
    parameters = arguments.get("parameters", arguments)
    if not isinstance(parameters, dict):
        raise TypeError("parameters debe ser un objeto JSON.")
    consulta = parameters["consulta"]
    if not isinstance(consulta, str) or not consulta.strip():
        raise TypeError("consulta debe ser texto no vacío.")
    limite = int(parameters.get("limite", 3))
    return consulta, limite


def formatear_respuesta(result: dict[str, Any]) -> str:
    respuesta_original = result["respuesta"]
    if respuesta_original.startswith("Respuesta detallada para la consulta sobre"):
        respuesta = (
            "La fuente identifica esta FAQ, pero no proporciona un detalle concreto "
            "para confirmar la respuesta. Para confirmarla, escribe a "
            "soporte@parachutesa.gt."
        )
    else:
        respuesta = respuesta_original
    respuesta_formateada = textwrap.fill(
        respuesta,
        width=88,
        initial_indent="  ",
        subsequent_indent="  ",
    )
    return (
        f"FAQ encontrada: {result['pregunta']}\n"
        f"Categoría: {result['categoria']}\n"
        f"Información disponible:\n{respuesta_formateada}\n"
        f"Referencia: {result['id']}"
    )


def responder(client: OpenAI, searcher: FAQSearcher, messages: list[dict[str, Any]], model: str) -> str:
    """Obliga la herramienta y devuelve la respuesta oficial recuperada."""
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[SEARCH_TOOL],
        tool_choice={"type": "function", "function": {"name": "buscar_faqs"}},
        temperature=0.2,
    )
    message = completion.choices[0].message
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
        response = (
            "No puedo responder esa pregunta con la información disponible. "
            "Puedes escribir a soporte@parachutesa.gt."
        )
        messages.append({"role": "assistant", "content": response})
        return response

    results_found: list[dict[str, Any]] = []
    should_refuse = False
    for tool_call in message.tool_calls:
        if tool_call.function.name != "buscar_faqs":
            output = json.dumps({"error": "Herramienta no permitida."})
            should_refuse = True
        else:
            try:
                consulta, limite = parsear_argumentos_busqueda(tool_call.function.arguments)
                results = buscar_combinado(
                    searcher, consulta, ultima_pregunta_usuario(messages), limite
                )
                output = serializar_resultados(results)
                results_found.extend(results)
                should_refuse = should_refuse or not results
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                output = json.dumps({"error": f"Argumentos inválidos: {error}"}, ensure_ascii=False)
                should_refuse = True
            except psycopg.Error:
                output = json.dumps(
                    {"error": "No fue posible consultar la base de conocimiento."}, ensure_ascii=False
                )
                should_refuse = True
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})

    if should_refuse or not results_found:
        response = (
            "No puedo responder esa pregunta con la información disponible. "
            "Puedes escribir a soporte@parachutesa.gt."
        )
    else:
        best_result = results_found[0]
        response = formatear_respuesta(best_result)
    messages.append({"role": "assistant", "content": response})
    return response


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
            print("\nAsistente: Buscando en la base de conocimiento...", flush=True)
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
