from langbot.bot import MemoryType
from langbot.cli import (
    build_parser,
    format_response,
    handle_command,
    print_welcome,
    run_cli,
)
import openai

from langbot.models import AssistanceResponse, Character, RequestType, ResponseChunk, ResponseComplete, ResponseStart


class StubBot:
    def __init__(self):
        self.character = Character.friendly
        self.memory = MemoryType.buffer
        self.history = ["u1", "a1"]
        self.stream_process_calls: list[str] = []
        self.cleared = False

    def clear_history(self):
        self.cleared = True
        self.history = []

    def switch_character(self, new_character: Character):
        self.character = new_character

    def switch_memory(self, new_memory: MemoryType):
        self.memory = new_memory

    def stream_process(self, text: str):
        self.stream_process_calls.append(text)
        yield ResponseStart(
            request_type=RequestType.question,
            confidence=0.923,
        )
        yield ResponseChunk(text=f"echo: {text}")
        yield ResponseComplete(response=AssistanceResponse(
            content=f"echo: {text}",
            request_type=RequestType.question,
            confidence=0.923,
            tokens_used=78,
        ))


def test_format_response_matches_spec():
    response = AssistanceResponse(
        content="Test reply",
        request_type=RequestType.task,
        confidence=0.923,
        tokens_used=78,
    )

    formatted = format_response(response)

    assert formatted == "[task] Test reply\nconfidence: 0.92 | tokens: ~78"


def test_build_parser_help_shows_plain_memory_names():
    help_text = build_parser().format_help()

    assert "--memory {buffer,summary}" in help_text


def test_handle_command_updates_character_memory_and_status():
    bot = StubBot()
    outputs: list[str] = []

    assert handle_command("/character pirate", bot, output_fn=outputs.append) is False
    assert handle_command("/memory summary", bot, output_fn=outputs.append) is False
    assert handle_command("/status", bot, output_fn=outputs.append) is False

    assert bot.character == Character.pirate
    assert bot.memory == MemoryType.summary
    assert outputs == [
        "✓ Характер изменён на: pirate",
        "✓ Память изменена на: summary",
        "Характер: pirate | Память: summary | Сообщений: 2",
    ]


def test_run_cli_processes_messages_and_exits_on_quit():
    bot = StubBot()
    outputs: list[str] = []
    inputs = iter(["hello", "/quit"])

    exit_code = run_cli(
        bot,
        input_fn=lambda _: next(inputs),
        output_fn=outputs.append,
        stream_output_fn=outputs.append,
    )

    assert exit_code == 0
    assert bot.stream_process_calls == ["hello"]
    assert outputs == [
        "[question] ",
        "echo: hello",
        "",
        "confidence: 0.92 | tokens: ~78",
        "",
        "До встречи.",
        "",
    ]


def test_run_cli_adds_blank_line_after_command_output():
    bot = StubBot()
    outputs: list[str] = []
    inputs = iter(["/character pirate", "/quit"])

    exit_code = run_cli(
        bot,
        input_fn=lambda _: next(inputs),
        output_fn=outputs.append,
        stream_output_fn=outputs.append,
    )

    assert exit_code == 0
    assert outputs == [
        "✓ Характер изменён на: pirate",
        "",
        "До встречи.",
        "",
    ]


def test_handle_command_rejects_unknown_values():
    bot = StubBot()
    outputs: list[str] = []

    handle_command("/character wizard", bot, output_fn=outputs.append)
    handle_command("/memory infinite", bot, output_fn=outputs.append)

    assert outputs == [
        "Неизвестный характер. Доступно: friendly, professional, sarcastic, pirate.",
        "Неизвестная стратегия памяти. Доступно: buffer, summary.",
    ]


def test_print_welcome_shows_russian_banner():
    bot = StubBot()
    outputs: list[str] = []

    print_welcome(bot, output_fn=outputs.append)

    assert outputs == [
        "🤖 Умный ассистент с характером",
        "Характер: friendly | Память: buffer",
        "────────────────────────────────",
        "",
    ]


def test_run_cli_prints_api_error_for_stream_failures():
    class FailingBot(StubBot):
        def stream_process(self, text: str):
            self.stream_process_calls.append(text)
            raise openai.APIConnectionError(request=None)
            yield

    bot = FailingBot()
    outputs: list[str] = []
    inputs = iter(["hello", "/quit"])

    exit_code = run_cli(
        bot,
        input_fn=lambda _: next(inputs),
        output_fn=outputs.append,
        stream_output_fn=outputs.append,
    )

    assert exit_code == 0
    assert outputs == [
        "Временная ошибка API: Connection error.",
        "",
        "До встречи.",
        "",
    ]
