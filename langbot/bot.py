import json
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    message_chunk_to_message,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RouterRunnable,
    Runnable,
    RunnablePassthrough,
    chain,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from langbot.models import *
from langbot.classifier import classifier_chain


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class MemoryType(Enum):
    buffer = 0
    summary = 1


class Bot:
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "{character}\n\n{instructions}"),
        MessagesPlaceholder(variable_name="history", optional=True),
        ("human", "{input}")
    ])

    def __init__(
        self,
        model: Runnable[Any, AIMessage],
        *,
        character: Character = Character.friendly,
        memory: MemoryType = MemoryType.buffer,
        classifier_model: Runnable[Any, AIMessage] | None = None
    ):
        self.character = character
        self.memory = memory

        self.main_model = model
        self.classifier_model = classifier_model or model

        self._history = InMemoryChatMessageHistory()

        self.classifier = classifier_chain(self.classifier_model)
        self.stream_response_chain: Runnable

        self._character_prompts = {
            Character(key): value 
            for key, value in _load_json("character_prompts.json").items()
        }
        self._request_type_instructions = {
            request_type: instructions
            for request_type, instructions in (
                (RequestType(key), value)
                for key, value in _load_json("request_prompts.json").items()
            )
        }
        self.assemble()

    def switch_character(self, new_character: Character):
        self.character = new_character
        self.assemble()

    def switch_memory(self, new_memory: MemoryType):
        self.memory = new_memory
        ...

    @property
    def history(self) -> list[BaseMessage]:
        return self._history.messages

    def clear_history(self):
        self._history.clear()

    def process(self, input: str):
        return self.chain.invoke({"input": input})

    def stream_process(self, input: str) -> Iterator[BotStreamEvent]:
        classification_result = self.classifier.invoke({"input": input})
        yield _start_event(classification_result)

        streamed_message: AIMessageChunk | None = None

        for chunk in self.stream_response_chain.stream({
            "input": input,
            "classification_result": classification_result,
        }):
            if not isinstance(chunk, AIMessageChunk):
                continue

            streamed_message = (
                chunk if streamed_message is None else streamed_message + chunk
            )

            text = _message_text(chunk)
            if text:
                yield ResponseChunk(text=text)

        response = _finalize_stream_response(
            input=input,
            streamed_message=streamed_message,
            classification_result=classification_result,
            history=self._history,
        )

        yield ResponseComplete(response=response)

    def assemble(self):
        router = self._request_type_router()

        self.stream_response_chain = (
            RunnablePassthrough.assign(history=lambda _: self.history)
            | _to_router_input
            | router
        )
        self.chain = (
            RunnablePassthrough.assign(classification_result=self.classifier)
            | RunnablePassthrough.assign(
                output=RunnableWithMessageHistory(
                    self.stream_response_chain,
                    lambda: self._history,
                    input_messages_key="input",
                    history_messages_key="history",
                )
            )
            | _to_assistance_response
        )

    def _request_type_router(self) -> Runnable:
        self.main_prompt = self.prompt_template.partial(
            character=self._character_prompts[self.character]
        )
        return RouterRunnable({
            request_type: self.main_prompt.partial(
                instructions=self._request_type_instructions[request_type]
            )
            | self.main_model
            for request_type in RequestType
        })


@chain
def _to_router_input(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": data["classification_result"]["parsed"].request_type,
        "input": data,
    }


@chain
def _to_assistance_response(data: dict[str, Any]) -> AssistanceResponse:
    return _assistance_response_from_message(
        output=data["output"],
        classification_result=data["classification_result"],
    )


def _load_json(filename: str) -> dict[str, str]:
    with (_PROMPTS_DIR / filename).open(encoding="utf-8") as f:
        return json.load(f)


def _tokens_used(message: AIMessage | AIMessageChunk | None) -> int:
    if message is None:
        return 0
    if message.usage_metadata is None:
        return 0
    return message.usage_metadata.get("total_tokens", 0)


def _usage_metadata(message: AIMessage | AIMessageChunk | None) -> Any:
    if message is None:
        return None
    return message.usage_metadata


def _start_event(classification_result: dict[str, Any]) -> ResponseStart:
    parsed = classification_result["parsed"]
    return ResponseStart(
        request_type=parsed.request_type,
        confidence=parsed.confidence,
    )


def _assistance_response_from_message(
    output: AIMessage | AIMessageChunk,
    classification_result: dict[str, Any],
) -> AssistanceResponse:
    return AssistanceResponse.model_validate({
        "content": _message_text(output),
        "request_type": classification_result["parsed"].request_type,
        "confidence": classification_result["parsed"].confidence,
        "tokens_used": (
            _tokens_used(classification_result["raw"])
            + _tokens_used(output)
        ),
    })


def _finalize_stream_response(
    *,
    input: str,
    streamed_message: AIMessageChunk | None,
    classification_result: dict[str, Any],
    history: InMemoryChatMessageHistory,
) -> AssistanceResponse:
    final_message = _final_message(streamed_message)
    response = _assistance_response_from_message(
        output=final_message,
        classification_result=classification_result,
    )
    history.add_messages([
        HumanMessage(content=input),
        AIMessage(
            content=response.content,
            usage_metadata=_usage_metadata(final_message),
        ),
    ])
    return response


def _final_message(chunk: AIMessageChunk | None) -> AIMessage:
    if chunk is None:
        return AIMessage(content="")
    message = message_chunk_to_message(chunk)
    if not isinstance(message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(message).__name__}")
    return message


def _message_text(message: AIMessage | AIMessageChunk) -> str:
    return str(message.text)
