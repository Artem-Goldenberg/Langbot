import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import PrivateAttr

from langbot.bot import Bot
from langbot.models import Character, Classification, RequestType


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
