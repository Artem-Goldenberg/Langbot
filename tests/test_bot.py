import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
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
class RecordingFakeModel(FakeListChatModel):
    _seen_calls: list[list] = PrivateAttr(default_factory=list)

    @property
    def seen_calls(self) -> list[list]:
        return self._seen_calls

    def get_num_tokens_from_messages(self, messages) -> int:
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

    assert [type(event) for event in events] == [
        ResponseStart,
        ResponseChunk,
        ResponseChunk,
        ResponseComplete,
    ]
    assert events[0].request_type == RequestType.question
    assert events[1].text == "first "
    assert events[2].text == "second"
    assert events[3].response.content == "first second"
    assert events[3].response.tokens_used == 16
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

    assert [type(event) for event in events] == [
        ResponseStart,
        ResponseChunk,
        ResponseChunk,
        ResponseComplete,
    ]
    assert events[1].text == "hello"
    assert events[2].text == " world"
    assert events[3].response.content == "hello world"
    assert [type(message) for message in bot.history] == [HumanMessage, AIMessage]
    assert bot.history[-1].content == "hello world"


def test_bot_buffer_memory_keeps_only_recent_messages():
    model = RecordingFakeModel(
        responses=["first response", "second response", "third response"]
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
        max_context_tokens=4,
    )

    bot.process("first input")
    bot.process("second input")
    bot.process("third input")

    assert [message.content for message in model.seen_calls[2]] == [
        f"{CHARACTER_PROMPTS[Character.friendly]}\n\n"
        f"{REQUEST_PROMPTS[RequestType.question]}",
        "second input",
        "second response",
        "third input",
    ]


def test_bot_summary_memory_uses_dedicated_summary_chain():
    model = RecordingFakeModel(responses=["first response", "second response"])
    summary_model = RecordingFakeModel(responses=["summary of first turn"])
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
    )
    assert [message.content for message in model.seen_calls[1][1:]] == ["second input"]
    assert [type(message) for message in bot.history] == [HumanMessage, AIMessage]
