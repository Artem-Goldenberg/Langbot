import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import PrivateAttr

from langbot.models import Character, RequestType
from langbot.router import routed_bot_chain


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "langbot" / "prompts"


def _load_json(filename: str) -> dict[str, str]:
    with (PROMPTS_DIR / filename).open(encoding="utf-8") as f:
        return json.load(f)


REQUEST_PROMPTS = _load_json("request_prompts.json")
CHARACTER_PROMPTS = _load_json("character_prompts.json")


class RecordingFakeModel(FakeListChatModel):
    _seen_messages: list | None = PrivateAttr(default=None)

    @property
    def seen_messages(self) -> list:
        assert self._seen_messages is not None
        return self._seen_messages

    def _call(self, *args, **kwargs):
        self._seen_messages = args[0]
        return super()._call(*args, **kwargs)


@pytest.mark.parametrize("request_type", list(RequestType))
def test_bot_chain_uses_exact_request_instructions_for_classification(
    request_type: RequestType,
):
    model = RecordingFakeModel(responses=["ok"])
    chain = routed_bot_chain(model, Character.professional)

    result = chain.invoke(
        {"key": str(request_type), "input": {"input": "test input"}}
    )

    assert result == "ok"
    assert model.seen_messages[0].content == (
        f"{CHARACTER_PROMPTS[Character.professional]}\n\n"
        f"{REQUEST_PROMPTS[request_type]}"
    )
    assert model.seen_messages[1].content == "test input"


@pytest.mark.parametrize("character", list(Character))
def test_bot_chain_uses_exact_character_prompt(character: Character):
    model = RecordingFakeModel(responses=["ok"])
    chain = routed_bot_chain(model, character)

    result = chain.invoke(
        {"key": str(RequestType.question), "input": {"input": "test input"}}
    )

    assert result == "ok"
    assert model.seen_messages[0].content == (
        f"{CHARACTER_PROMPTS[character]}\n\n"
        f"{REQUEST_PROMPTS[RequestType.question]}"
    )
    assert model.seen_messages[1].content == "test input"


def test_bot_chain_raises_for_unexpected_character():
    model = RecordingFakeModel(responses=["ok"])

    with pytest.raises(KeyError):
        routed_bot_chain(model, "unexpected")  # type: ignore[arg-type]
