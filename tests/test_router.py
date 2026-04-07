import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from pydantic import PrivateAttr
from langchain_core.runnables import RunnableLambda

from langbot.models import Character, Classification, RequestType
from langbot.bot import Bot


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "langbot" / "prompts"


def _load_json(filename: str) -> dict[str, str]:
    with (PROMPTS_DIR / filename).open(encoding="utf-8") as f:
        return json.load(f)


REQUEST_PROMPTS = _load_json("request_prompts.json")
CHARACTER_PROMPTS = _load_json("character_prompts.json")


class RecordingFakeModel(FakeListChatModel):
    _seen_calls: list[list] = PrivateAttr(default_factory=list)

    @property
    def seen_messages(self) -> list:
        assert self._seen_calls
        return self._seen_calls[-1]

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


@pytest.mark.parametrize("request_type", list(RequestType))
def test_bot_process_uses_exact_request_instructions_for_classification(
    request_type: RequestType,
):
    model = RecordingFakeModel(responses=["ok"])
    classifier_model = FixedClassificationModel(
        Classification(
            request_type=request_type,
            confidence=0.99,
            reasoning="fixed",
        )
    )
    bot = Bot(
        model,
        character=Character.professional,
        classifier_model=classifier_model,
    )

    result = bot.process("test input")

    assert result.content == "ok"
    assert result.request_type == request_type
    assert model.seen_messages[0].content == (
        f"{CHARACTER_PROMPTS[Character.professional]}\n\n"
        f"{REQUEST_PROMPTS[request_type]}"
    )
    assert model.seen_messages[1].content == "test input"


@pytest.mark.parametrize("character", list(Character))
def test_bot_process_uses_exact_character_prompt(character: Character):
    model = RecordingFakeModel(responses=["ok"])
    classifier_model = FixedClassificationModel(
        Classification(
            request_type=RequestType.question,
            confidence=0.99,
            reasoning="fixed",
        )
    )
    bot = Bot(model, character=character, classifier_model=classifier_model)

    result = bot.process("test input")

    assert result.content == "ok"
    assert result.request_type == RequestType.question
    assert model.seen_messages[0].content == (
        f"{CHARACTER_PROMPTS[character]}\n\n"
        f"{REQUEST_PROMPTS[RequestType.question]}"
    )
    assert model.seen_messages[1].content == "test input"


def test_bot_raises_for_unexpected_character():
    model = RecordingFakeModel(responses=["ok"])

    with pytest.raises(KeyError):
        Bot(model, character="unexpected")  # type: ignore[arg-type]
