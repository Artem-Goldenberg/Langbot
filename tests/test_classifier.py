from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from langbot.classifier import classifier_chain
from langbot.models import Classification, RequestType


class StructuredOutputFakeModel(FakeListChatModel):
    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ):
        schema_name = schema.__name__ if isinstance(schema, type) else "dict"
        return RunnableLambda(
            lambda _: {
                "raw": AIMessage(
                    content="",
                    usage_metadata={
                        "input_tokens": 3,
                        "output_tokens": 2,
                        "total_tokens": 5,
                    },
                ),
                "parsed": Classification(
                    request_type=RequestType.question,
                    confidence=0.99,
                    reasoning=schema_name,
                ),
            }
        )


class ExplodingFakeModel(FakeListChatModel):
    def _call(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_classifier_uses_structured_output_when_supported():
    model = StructuredOutputFakeModel(responses=["unused"])

    result = classifier_chain(model).invoke({"input": "Почему небо голубое?"})

    assert result["parsed"] == Classification(
        request_type=RequestType.question,
        confidence=0.99,
        reasoning="Classification",
    )
    assert result["raw"].usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }


def test_classifier_returns_unknown_when_parser_cannot_parse():
    model = FakeListChatModel(responses=["это вообще не json"])

    result = classifier_chain(model).invoke({"input": "Але?"})

    assert result["parsed"] == Classification(
        request_type=RequestType.unknown,
        confidence=0.5,
        reasoning="Ошибка парсинга ответа модели",
    )
    assert result["raw"].content == "это вообще не json"


def test_classifier_returns_unknown_when_structured_output_cannot_parse():
    class BrokenStructuredOutputModel(StructuredOutputFakeModel):
        def with_structured_output(
            self,
            schema: dict[str, Any] | type,
            *,
            include_raw: bool = False,
            **kwargs: Any,
        ):
            return RunnableLambda(
                lambda _: {
                    "raw": AIMessage(content=""),
                    "parsed": None,
                }
            )

    model = BrokenStructuredOutputModel(responses=["unused"])

    result = classifier_chain(model).invoke({"input": "Але?"})

    assert result["parsed"] == Classification(
        request_type=RequestType.unknown,
        confidence=0.5,
        reasoning="Ошибка парсинга ответа модели",
    )


def test_classifier_does_not_swallow_model_errors():
    model = ExplodingFakeModel(responses=["unused"])

    with pytest.raises(RuntimeError, match="boom"):
        classifier_chain(model).invoke({"input": "Але?"})
