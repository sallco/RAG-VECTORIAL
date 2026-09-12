"""Punto de entrada para cargar el corpus o iniciar el agente conversacional."""

from __future__ import annotations

import argparse
from typing import Sequence

import agent
import load_faqs


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proyecto RAG de FAQs para Parachute S.A.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("load", help="Genera embeddings y carga el corpus.")
    commands.add_parser("chat", help="Inicia el agente conversacional.")
    parsed_arguments, remaining_arguments = parser.parse_known_args(arguments)
    if parsed_arguments.command == "load":
        parsed_arguments.loader_arguments = remaining_arguments
    elif remaining_arguments:
        parser.error(f"argumentos no reconocidos: {' '.join(remaining_arguments)}")
    return parsed_arguments


def main(arguments: Sequence[str] | None = None) -> None:
    parsed_arguments = parse_arguments(arguments)
    if parsed_arguments.command == "load":
        load_faqs.main(parsed_arguments.loader_arguments)
    elif parsed_arguments.command == "chat":
        agent.main()


if __name__ == "__main__":
    main()
