import json
from pathlib import Path
from typing import Any, Sequence, cast

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.runnables import RunnableLambda
from pydantic import PrivateAttr

from langbot.bot import Bot, MemoryType
from langbot.models import (
    Character,
    Classification,
    RequestType,
    ResponseChunk,
    ResponseComplete,
    ResponseStart,
)


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "langbot" / "prompts"


def _load_json(filename: str) -> dict[str, str]:
    with (PROMPTS_DIR / filename).open(encoding="utf-8") as f:
        return json.load(f)


REQUEST_PROMPTS = _load_json("request_prompts.json")
CHARACTER_PROMPTS = _load_json("character_prompts.json")


def _entities_suffix(entities: dict[str, object]) -> str:
    return (
        "\n\nИзвестные данные о пользователе:\n"
        f"{json.dumps(entities, ensure_ascii=False, indent=2, sort_keys=True)}"
    )


def _known_entities_prompt(entities: dict[str, object]) -> str:
    return f"Уже известные сущности:\n{json.dumps(entities, ensure_ascii=False, indent=2, sort_keys=True)}"

class RecordingFakeModel(FakeListChatModel):
    _seen_calls: list[list] = PrivateAttr(default_factory=list)

    @property
    def seen_calls(self) -> list[list]:
        return self._seen_calls

    def get_num_tokens_from_messages(
        self,
        messages: list[BaseMessage],
        tools: Sequence[Any] | None = None,
    ) -> int:
        return sum(len(str(message.content).split()) for message in messages)

    def _call(self, *args, **kwargs):
        self._seen_calls.append(list(args[0]))
        return super()._call(*args, **kwargs)


class FixedClassificationModel(FakeListChatModel):
    def __init__(self, classification: Classification):
        super().__init__(responses=["unused"])
        self._classification = classification

    def with_structured_output(self, *args, **kwargs):
        return RunnableLambda(
            lambda _: {
                "raw": AIMessage(content=""),
                "parsed": self._classification,
            }
        )


class UsageClassificationModel(FixedClassificationModel):
    def __init__(self, classification: Classification, *, total_tokens: int):
        super().__init__(classification)
        self._total_tokens = total_tokens

    def with_structured_output(self, *args, **kwargs):
        return RunnableLambda(
            lambda _: {
                "raw": AIMessage(
                    content="",
                    usage_metadata={
                        "input_tokens": self._total_tokens - 1,
                        "output_tokens": 1,
                        "total_tokens": self._total_tokens,
                    },
                ),
                "parsed": self._classification,
            }
        )


class UsageFakeModel(RecordingFakeModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._seen_calls.append(list(messages))
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0

        return ChatResult(generations=[
            ChatGeneration(
                message=AIMessage(
                    content=response,
                    usage_metadata={
                        "input_tokens": 7,
                        "output_tokens": 5,
                        "total_tokens": 12,
                    },
                )
            )
        ])


class StreamingUsageFakeModel(RecordingFakeModel):
    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self._seen_calls.append(list(messages))
        yield ChatGenerationChunk(message=AIMessageChunk(content="first "))
        yield ChatGenerationChunk(message=AIMessageChunk(
            content="second",
            usage_metadata={
                "input_tokens": 7,
                "output_tokens": 5,
                "total_tokens": 12,
            },
        ))


class BlockStreamingFakeModel(RecordingFakeModel):
    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self._seen_calls.append(list(messages))
        yield ChatGenerationChunk(message=AIMessageChunk(
            content=[{"type": "text", "text": "hello"}]
        ))
        yield ChatGenerationChunk(message=AIMessageChunk(
            content=[{"type": "text", "text": " world"}]
        ))


def test_bot_raises_for_unexpected_character():
    model = RecordingFakeModel(responses=["ok"])

    with pytest.raises(KeyError):
        Bot(model, character="unexpected")  # type: ignore[arg-type]


def test_bot_process_includes_prior_history():
    model = RecordingFakeModel(responses=["first response", "second response"])
    classifier_model = FixedClassificationModel(
        Classification(
            request_type=RequestType.question,
            confidence=0.99,
            reasoning="fixed",
        )
    )
    bot = Bot(
        model,
        character=Character.professional,
        classifier_model=classifier_model,
    )

    first = bot.process("first input")
    second = bot.process("second input")

    assert first.content == "first response"
    assert second.content == "second response"
    assert [message.content for message in model.seen_calls[1]] == [
        f"{CHARACTER_PROMPTS[Character.professional]}\n\n"
        f"{REQUEST_PROMPTS[RequestType.question]}",
        "first input",
        "first response",
        "second input",
    ]
    assert [type(message) for message in bot.history] == [
        HumanMessage,
        AIMessage,
        HumanMessage,
        AIMessage,
    ]
    assert [message.content for message in bot.history] == [
        "first input",
        "first response",
        "second input",
        "second response",
    ]


def test_bot_process_reports_model_token_usage():
    model = UsageFakeModel(responses=["ok"])
    classifier_model = UsageClassificationModel(
        Classification(
            request_type=RequestType.question,
            confidence=0.99,
            reasoning="fixed",
        ),
        total_tokens=4,
    )
    bot = Bot(model, classifier_model=classifier_model)

    result = bot.process("test input")

    assert result.content == "ok"
    assert result.tokens_used == 16


def test_bot_stream_process_emits_events_and_updates_history_on_completion():
    model = StreamingUsageFakeModel(responses=["unused"])
    classifier_model = UsageClassificationModel(
        Classification(
            request_type=RequestType.question,
            confidence=0.99,
            reasoning="fixed",
        ),
        total_tokens=4,
    )
    bot = Bot(
        model,
        character=Character.professional,
        classifier_model=classifier_model,
    )

    events = list(bot.stream_process("stream input"))
    start = cast(ResponseStart, events[0])
    first_chunk = cast(ResponseChunk, events[1])
    second_chunk = cast(ResponseChunk, events[2])
    complete = cast(ResponseComplete, events[3])

    assert [type(event) for event in events] == [
        ResponseStart,
        ResponseChunk,
        ResponseChunk,
        ResponseComplete,
    ]
    assert start.request_type == RequestType.question
    assert first_chunk.text == "first "
    assert second_chunk.text == "second"
    assert complete.response.content == "first second"
    assert complete.response.tokens_used == 16
    assert [message.content for message in model.seen_calls[0]] == [
        f"{CHARACTER_PROMPTS[Character.professional]}\n\n"
        f"{REQUEST_PROMPTS[RequestType.question]}",
        "stream input",
    ]
    assert [type(message) for message in bot.history] == [HumanMessage, AIMessage]
    assert [message.content for message in bot.history] == [
        "stream input",
        "first second",
    ]


def test_bot_stream_process_does_not_update_history_when_stream_fails():
    class FailingStreamModel(RecordingFakeModel):
        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            self._seen_calls.append(list(messages))
            yield ChatGenerationChunk(message=AIMessageChunk(content="partial"))
            raise RuntimeError("boom")

    model = FailingStreamModel(responses=["unused"])
    classifier_model = FixedClassificationModel(
        Classification(
            request_type=RequestType.question,
            confidence=0.99,
            reasoning="fixed",
        )
    )
    bot = Bot(model, classifier_model=classifier_model)

    with pytest.raises(RuntimeError, match="boom"):
        list(bot.stream_process("stream input"))

    assert bot.history == []


def test_bot_stream_process_supports_block_content():
    model = BlockStreamingFakeModel(responses=["unused"])
    classifier_model = FixedClassificationModel(
        Classification(
            request_type=RequestType.question,
            confidence=0.99,
            reasoning="fixed",
        )
    )
    bot = Bot(model, classifier_model=classifier_model)

    events = list(bot.stream_process("stream input"))
    first_chunk = cast(ResponseChunk, events[1])
    second_chunk = cast(ResponseChunk, events[2])
    complete = cast(ResponseComplete, events[3])

    assert [type(event) for event in events] == [
        ResponseStart,
        ResponseChunk,
        ResponseChunk,
        ResponseComplete,
    ]
    assert first_chunk.text == "hello"
    assert second_chunk.text == " world"
    assert complete.response.content == "hello world"
    assert [type(message) for message in bot.history] == [HumanMessage, AIMessage]
    assert bot.history[-1].content == "hello world"


def test_bot_buffer_memory_keeps_only_recent_messages():
    model = RecordingFakeModel(
        responses=["first response", "second response", "third response"]
    )
    entity_model = RecordingFakeModel(
        responses=['{"user_name":"Alice","home_city":"Paris"}']
    )
    classifier_model = FixedClassificationModel(
        Classification(
            request_type=RequestType.question,
            confidence=0.99,
            reasoning="fixed",
        )
    )
    bot = Bot(
        model,
        memory_type=MemoryType.buffer,
        classifier_model=classifier_model,
        entity_model=entity_model,
        max_context_tokens=4,
    )

    bot.process("first input")
    bot.process("second input")
    bot.process("third input")

    assert entity_model.seen_calls[0][0].content.endswith(_known_entities_prompt({}))
    assert [message.content for message in entity_model.seen_calls[0][1:]] == [
        "first input",
        "first response",
        "second input",
        "second response",
    ]
    assert [message.content for message in model.seen_calls[2]] == [
        f"{CHARACTER_PROMPTS[Character.friendly]}\n\n"
        f"{REQUEST_PROMPTS[RequestType.question]}"
        f"{_entities_suffix({'user_name': 'Alice', 'home_city': 'Paris'})}",
        "second input",
        "second response",
        "third input",
    ]
    assert bot.entities == {
        "user_name": "Alice",
        "home_city": "Paris",
    }


def test_bot_buffer_memory_supports_fallback_wrapped_model():
    primary_model = RecordingFakeModel(
        responses=["first response", "second response", "third response"]
    )
    fallback_model = RecordingFakeModel(responses=["unused fallback"])
    classifier_model = FixedClassificationModel(
        Classification(
            request_type=RequestType.question,
            confidence=0.99,
            reasoning="fixed",
        )
    )
    bot = Bot(
        primary_model.with_fallbacks([fallback_model]),
        memory_type=MemoryType.buffer,
        classifier_model=classifier_model,
        max_context_tokens=4,
    )

    bot.process("first input")
    bot.process("second input")
    response = bot.process("third input")

    assert response.content
    assert [message.content for message in primary_model.seen_calls[3]] == [
        f"{CHARACTER_PROMPTS[Character.friendly]}\n\n"
        f"{REQUEST_PROMPTS[RequestType.question]}",
        "second input",
        "second response",
        "third input",
    ]


def test_bot_summary_memory_uses_dedicated_summary_chain():
    model = RecordingFakeModel(responses=["first response", "second response"])
    summary_model = RecordingFakeModel(responses=["summary of first turn"])
    entity_model = RecordingFakeModel(
        responses=['{"user_name":"Alice","preferences":{"drink":"tea"}}']
    )
    classifier_model = FixedClassificationModel(
        Classification(
            request_type=RequestType.question,
            confidence=0.99,
            reasoning="fixed",
        )
    )
    bot = Bot(
        model,
        memory_type=MemoryType.summary,
        classifier_model=classifier_model,
        summary_model=summary_model,
        entity_model=entity_model,
        max_context_tokens=3,
    )

    bot.process("first input")
    bot.process("second input")

    assert [message.content for message in summary_model.seen_calls[0]] == [
        "Ты сжимаешь историю диалога в короткую рабочую сводку для ассистента. "
        "Сохраняй факты, задачи, предпочтения пользователя, обещания и открытые "
        "вопросы. Пиши кратко, без вводных фраз и без выдуманных деталей.",
        "first input",
        "first response",
    ]
    assert model.seen_calls[1][0].content == (
        f"{CHARACTER_PROMPTS[Character.friendly]}\n\n"
        f"{REQUEST_PROMPTS[RequestType.question]}\n\n"
        "Краткая сводка предыдущего диалога:\nsummary of first turn"
        f"{_entities_suffix({'user_name': 'Alice', 'preferences': {'drink': 'tea'}})}"
    )
    assert [message.content for message in model.seen_calls[1][1:]] == ["second input"]
    assert entity_model.seen_calls[0][0].content.endswith(_known_entities_prompt({}))
    assert [message.content for message in entity_model.seen_calls[0][1:]] == [
        "first input",
        "first response",
    ]
    assert [type(message) for message in bot.history] == [HumanMessage, AIMessage]
    assert bot.entities == {
        "user_name": "Alice",
        "preferences": {"drink": "tea"},
    }


def test_bot_process_writes_trace_log(tmp_path: Path):
    model = UsageFakeModel(responses=["ok"])
    classifier_model = UsageClassificationModel(
        Classification(
            request_type=RequestType.question,
            confidence=0.99,
            reasoning="fixed",
        ),
        total_tokens=4,
    )
    log_path = tmp_path / "trace.log"
    bot = Bot(
        model,
        classifier_model=classifier_model,
        log_path=str(log_path),
    )

    response = bot.process("test input")

    assert response.content == "ok"
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "Langbot Session Trace" in log_text
    assert "Started" in log_text
    assert "\"user_input\": \"test input\"" in log_text
    assert "response Model Response" in log_text
    assert "\"content\": \"ok\"" in log_text
    assert "\"total_tokens\": 12" in log_text
    assert "Completed" in log_text


def test_bot_logs_memory_events_to_trace_file(tmp_path: Path):
    model = RecordingFakeModel(responses=["first response", "second response"])
    summary_model = RecordingFakeModel(responses=["summary of first turn"])
    entity_model = RecordingFakeModel(
        responses=['{"user_name":"Alice","home_city":"Paris"}']
    )
    classifier_model = FixedClassificationModel(
        Classification(
            request_type=RequestType.question,
            confidence=0.99,
            reasoning="fixed",
        )
    )

    log_path = tmp_path / "trace.log"
    bot = Bot(
        model,
        memory_type=MemoryType.summary,
        classifier_model=classifier_model,
        summary_model=summary_model,
        entity_model=entity_model,
        max_context_tokens=3,
        log_path=str(log_path),
    )

    bot.process("first input")
    bot.process("second input")
    bot.switch_memory(MemoryType.buffer)
    bot.clear_history()

    log_text = log_path.read_text(encoding="utf-8")
    assert "Summary Memory Triggered" in log_text
    assert "\"new_summary\": \"summary of first turn\"" in log_text
    assert "Entity Memory Triggered" in log_text
    assert "\"source\": \"history_reduction\"" in log_text
    assert "entity Model Response" in log_text
    assert "Entity Memory Updated" in log_text
    assert "\"extracted_entities\": {" in log_text
    assert "\"home_city\": \"Paris\"" in log_text
    assert "Memory Strategy Switched" in log_text
    assert "\"new_memory\": \"buffer\"" in log_text
    assert "History Cleared" in log_text
    assert "\"previous_summary\": \"summary of first turn\"" in log_text


def test_bot_creates_trace_file_at_startup(tmp_path: Path):
    log_path = tmp_path / "trace.log"

    Bot(RecordingFakeModel(responses=["ok"]), log_path=str(log_path))

    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "Langbot Session Trace" in log_text
    assert "Session Initialized" in log_text


def test_bot_clear_history_extracts_entities_and_keeps_them_after_clear():
    model = RecordingFakeModel(responses=["first response", "second response"])
    entity_model = RecordingFakeModel(
        responses=['{"user_name":"Anna","home_city":"Berlin"}']
    )
    classifier_model = FixedClassificationModel(
        Classification(
            request_type=RequestType.question,
            confidence=0.99,
            reasoning="fixed",
        )
    )
    bot = Bot(
        model,
        classifier_model=classifier_model,
        entity_model=entity_model,
    )

    bot.process("Меня зовут Анна")
    bot.clear_history()
    response = bot.process("Кто я?")

    assert response.content == "second response"
    assert [type(message) for message in bot.history] == [HumanMessage, AIMessage]
    assert [message.content for message in bot.history] == ["Кто я?", "second response"]
    assert bot.entities == {
        "user_name": "Anna",
        "home_city": "Berlin",
    }
    assert entity_model.seen_calls[0][0].content.endswith(_known_entities_prompt({}))
    assert [message.content for message in entity_model.seen_calls[0][1:]] == [
        "Меня зовут Анна",
        "first response",
    ]
    assert model.seen_calls[1][0].content == (
        f"{CHARACTER_PROMPTS[Character.friendly]}\n\n"
        f"{REQUEST_PROMPTS[RequestType.question]}"
        f"{_entities_suffix({'user_name': 'Anna', 'home_city': 'Berlin'})}"
    )


def test_bot_entity_memory_includes_known_entities_and_deep_merges_updates():
    model = RecordingFakeModel(responses=["first response", "second response"])
    entity_model = RecordingFakeModel(
        responses=[
            '{"profile":{"name":"Anna","city":"Berlin"}}',
            '{"profile":{"likes":["tea"]},"timezone":"Europe/Berlin"}',
        ]
    )
    classifier_model = FixedClassificationModel(
        Classification(
            request_type=RequestType.question,
            confidence=0.99,
            reasoning="fixed",
        )
    )
    bot = Bot(
        model,
        classifier_model=classifier_model,
        entity_model=entity_model,
    )

    bot.process("Меня зовут Анна")
    bot.clear_history()
    bot.process("Я люблю чай")
    bot.clear_history()

    assert entity_model.seen_calls[1][0].content.endswith(_known_entities_prompt({
        "profile": {
            "city": "Berlin",
            "name": "Anna",
        }
    }))
    assert [message.content for message in entity_model.seen_calls[1][1:]] == [
        "Я люблю чай",
        "second response",
    ]
    assert bot.entities == {
        "profile": {
            "name": "Anna",
            "city": "Berlin",
            "likes": ["tea"],
        },
        "timezone": "Europe/Berlin",
    }


def test_bot_ignores_entity_parser_failures():
    model = RecordingFakeModel(responses=["first response", "second response"])
    entity_model = RecordingFakeModel(responses=["not json at all"])
    classifier_model = FixedClassificationModel(
        Classification(
            request_type=RequestType.question,
            confidence=0.99,
            reasoning="fixed",
        )
    )
    bot = Bot(
        model,
        classifier_model=classifier_model,
        entity_model=entity_model,
        max_context_tokens=3,
    )

    bot.process("first input")
    bot.clear_history()
    response = bot.process("second input")

    assert response.content == "second response"
    assert bot.entities == {}
    assert [type(message) for message in bot.history] == [HumanMessage, AIMessage]
    assert [message.content for message in bot.history] == [
        "second input",
        "second response",
    ]
