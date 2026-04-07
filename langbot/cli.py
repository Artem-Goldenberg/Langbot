"""CLI entry point for the bot project."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import openai
from dotenv import load_dotenv
from langchain_community.cache import SQLAlchemyCache
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine

from langbot.bot import Bot, MemoryType
from langbot.models import AssistanceResponse, Character, ResponseChunk, ResponseComplete, ResponseStart

RETRIABLE_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]
StreamOutputFn = Callable[[str], None]

HELP_TEXT = """Команды:
/clear - очистить историю диалога
/character <name> - сменить характер (friendly, professional, sarcastic, pirate)
/memory <strategy> - сменить стратегию памяти (buffer, summary)
/status - показать текущие настройки и количество сообщений
/help - показать эту справку
/quit - выйти
"""

DEFAULT_CACHE_DB = ".cache/langbot.db"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Интерактивный CLI для langbot.",
        add_help=False,
    )
    memory_choices = [memory.value for memory in MemoryType]
    parser._optionals.title = "Опции"
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="показать эту справку и выйти",
    )
    parser.add_argument(
        "--character",
        type=Character,
        choices=list(Character),
        default=Character.friendly,
        help="Характер ассистента.",
    )
    parser.add_argument(
        "--memory",
        type=_parse_memory_type,
        choices=memory_choices,
        default=MemoryType.buffer,
        metavar="{" + ",".join(memory_choices) + "}",
        help="Стратегия памяти.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="Имя модели ChatOpenAI.",
    )
    parser.add_argument(
        "--cache-db",
        default=DEFAULT_CACHE_DB,
        help=f"Путь к SQLite-файлу для LLM-кеша. По умолчанию: {DEFAULT_CACHE_DB}.",
    )
    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    configure_llm_cache(args.cache_db)
    bot = create_bot(
        model_name=args.model,
        character=args.character,
        memory_type=args.memory,
    )
    print_welcome(bot)
    return run_cli(bot)


def create_bot(
    *,
    model_name: str,
    character: Character,
    memory_type: MemoryType,
) -> Bot:
    model = ChatOpenAI(model=model_name).with_retry(
        retry_if_exception_type=RETRIABLE_EXCEPTIONS
    )
    return Bot(model, character=character, memory_type=memory_type)


def configure_llm_cache(cache_db: str | None) -> None:
    if not cache_db:
        return

    db_path = Path(cache_db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    set_llm_cache(SQLAlchemyCache(engine))


def run_cli(
    bot: Bot,
    *,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
    stream_output_fn: StreamOutputFn | None = None,
) -> int:
    stream_output = stream_output_fn or _default_stream_output

    while True:
        try:
            user_input = input_fn("> ").strip()
        except EOFError:
            output_fn("До встречи.")
            return 0
        except KeyboardInterrupt:
            output_fn("\nДо встречи.")
            return 0

        if not user_input:
            continue

        if user_input.startswith("/"):
            should_exit = handle_command(user_input, bot, output_fn=output_fn)
            output_fn("")
            if should_exit:
                return 0
            continue

        try:
            response = _stream_response(
                bot,
                user_input,
                output_fn=output_fn,
                stream_output_fn=stream_output,
            )
        except RETRIABLE_EXCEPTIONS as exc:
            output_fn(f"Временная ошибка API: {exc}")
            output_fn("")
            continue

        output_fn("")


def handle_command(command_line: str, bot: Bot, *, output_fn: OutputFn) -> bool:
    command, _, argument = command_line.partition(" ")
    argument = argument.strip()

    if command == "/clear":
        bot.clear_history()
        output_fn("✓ История очищена")
        return False

    if command == "/character":
        if not argument:
            output_fn(_invalid_character_message())
            return False
        try:
            character = Character(argument)
        except ValueError:
            output_fn(_invalid_character_message())
            return False
        bot.switch_character(character)
        output_fn(f"✓ Характер изменён на: {character.value}")
        return False

    if command == "/memory":
        if not argument:
            output_fn(_invalid_memory_message())
            return False
        try:
            memory = _parse_memory_type(argument)
        except ValueError:
            output_fn(_invalid_memory_message())
            return False
        bot.switch_memory(memory)
        output_fn(f"✓ Память изменена на: {memory.value}")
        return False

    if command == "/status":
        output_fn(
            f"Характер: {bot.character.value} | "
            f"Память: {bot.memory_type.value} | "
            f"Сообщений: {len(bot.history)}"
        )
        return False

    if command == "/help":
        print_help(output_fn=output_fn)
        return False

    if command == "/quit":
        output_fn("До встречи.")
        return True

    output_fn("Неизвестная команда. Используйте /help.")
    return False


def format_response(response: AssistanceResponse) -> str:
    return (
        f"[{response.request_type.value}] {response.content}\n"
        f"confidence: {response.confidence:.2f} | tokens: ~{response.tokens_used}"
    )


def _stream_response(
    bot: Bot,
    user_input: str,
    *,
    output_fn: OutputFn,
    stream_output_fn: StreamOutputFn,
) -> AssistanceResponse:
    response: AssistanceResponse | None = None

    for event in bot.stream_process(user_input):
        if isinstance(event, ResponseStart):
            stream_output_fn(f"[{event.request_type.value}] ")
            continue

        if isinstance(event, ResponseChunk):
            stream_output_fn(event.text)
            continue

        if isinstance(event, ResponseComplete):
            response = event.response
            output_fn("")
            output_fn(
                f"confidence: {response.confidence:.2f} | tokens: ~{response.tokens_used}"
            )

    if response is None:
        raise RuntimeError("Bot stream ended without completion event")

    return response


def print_help(*, output_fn: OutputFn = print) -> None:
    output_fn(HELP_TEXT.rstrip())


def print_welcome(bot: Bot, *, output_fn: OutputFn = print) -> None:
    output_fn("🤖 Умный ассистент с характером")
    output_fn(f"Характер: {bot.character.value} | Память: {bot.memory_type.value}")
    output_fn("────────────────────────────────")
    output_fn("")


def _parse_memory_type(value: str) -> MemoryType:
    try:
        return MemoryType(value)
    except ValueError as exc:
        raise ValueError(value) from exc


def _invalid_character_message() -> str:
    values = ", ".join(character.value for character in Character)
    return f"Неизвестный характер. Доступно: {values}."


def _invalid_memory_message() -> str:
    values = ", ".join(memory.value for memory in MemoryType)
    return f"Неизвестная стратегия памяти. Доступно: {values}."


def _default_stream_output(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()
