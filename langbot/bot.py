import json
from collections.abc import Iterator
from enum import StrEnum, auto
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
from langchain_core.runnables import RouterRunnable, Runnable, RunnablePassthrough, chain

from langbot.classifier import classifier_chain
from langbot.memory import history_fits_context, summary_chain, trim_history
from langbot.models import *


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_MAX_CONTEXT_TOKENS = 2_048


class MemoryType(StrEnum):
    buffer = auto()
    summary = auto()


class Bot:
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "{character}\n\n{instructions}{summary}"),
        MessagesPlaceholder(variable_name="history", optional=True),
        ("human", "{input}"),
    ])

    def __init__(
        self,
        model: Runnable[Any, AIMessage],
        *,
        character: Character = Character.friendly,
        memory_type: MemoryType = MemoryType.buffer,
        classifier_model: Runnable[Any, AIMessage] | None = None,
        summary_model: Runnable[Any, AIMessage] | None = None,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    ):
        self.character = character
        self.memory_type = memory_type

        self.main_model = model
        self.classifier_model = classifier_model or model
        self.summary_model = summary_model or model

        self.max_context_tokens = max_context_tokens
        self.summary_chain = summary_chain(self.summary_model)
        self._summary: str | None = None

        self._history = InMemoryChatMessageHistory()
        self.classifier = classifier_chain(self.classifier_model)

        self.main_prompt: ChatPromptTemplate
        self.stream_response_chain: Runnable
        self.chain: Runnable

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
        self.memory_type = new_memory

    @property
    def history(self) -> list[BaseMessage]:
        return self._history.messages

    def clear_history(self):
        self._history.clear()
        self._set_summary(None)

    def process(self, input: str) -> AssistanceResponse:
        chain_input = self._prepare_chain_input(input)
        result = self.chain.invoke(chain_input)
        self._history.add_messages([
            HumanMessage(content=input),
            result["output"],
        ])
        return _to_assistance_response(result)

    def stream_process(self, input: str) -> Iterator[BotStreamEvent]:
        chain_input = self._prepare_chain_input(input)

        classification_result = self.classifier.invoke({"input": input})
        yield _start_event(classification_result)

        streamed_message: AIMessageChunk | None = None

        for chunk in self.stream_response_chain.stream({
            **chain_input,
            "classification_result": classification_result,
        }):
            if not isinstance(chunk, AIMessageChunk):
                continue
            streamed_message = (
                chunk if streamed_message is None else streamed_message + chunk
            )
            text = str(chunk.text)
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
        self._refresh_main_prompt()
        router = self._request_type_router()
        self.stream_response_chain = _to_router_input | router
        self.chain = (
            RunnablePassthrough.assign(classification_result=self.classifier)
            | RunnablePassthrough.assign(output=self.stream_response_chain)
        )

    def _request_type_router(self) -> Runnable:
        return RouterRunnable({
            request_type: self.main_prompt.partial(
                instructions=self._request_type_instructions[request_type]
            )
            | self.main_model
            for request_type in RequestType
        })

    def _prepare_chain_input(self, input: str) -> dict[str, Any]:
        self._history.messages = self._reduced_history()
        return {
            "input": input,
            "history": self.history,
        }

    def _reduced_history(self) -> list[BaseMessage]:
        history = self.history
        if history_fits_context(
            history,
            model=self.main_model,
            max_tokens=self.max_context_tokens,
        ):
            return history

        match self.memory_type:
            case MemoryType.buffer:
                return trim_history(
                    history,
                    max_tokens=self.max_context_tokens,
                    model=self.main_model,
                )
            case MemoryType.summary:
                self._set_summary(self.summary_chain.invoke({"history": history}))
                return []

    def _set_summary(self, summary: str | None):
        self._summary = summary
        self.assemble()

    def _refresh_main_prompt(self):
        self.main_prompt = self.prompt_template.partial(
            character=self._character_prompts[self.character],
            summary=_summary_suffix(self._summary),
        )


def _summary_suffix(summary: str | None) -> str:
    if summary is None:
        return ""
    return f"\n\nКраткая сводка предыдущего диалога:\n{summary}"


@chain
def _to_router_input(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": data["classification_result"]["parsed"].request_type,
        "input": data,
    }


def _to_assistance_response(data: dict[str, Any]) -> AssistanceResponse:
    return _assistance_response_from_message(
        output=data["output"],
        classification_result=data["classification_result"],
    )


def _load_json(filename: str) -> dict[str, str]:
    with (_PROMPTS_DIR / filename).open(encoding="utf-8") as f:
        return json.load(f)


def _tokens_used(message: AIMessage | AIMessageChunk | None) -> int:
    if message is None or message.usage_metadata is None:
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
    return AssistanceResponse(
        content=str(output.text),
        request_type=classification_result["parsed"].request_type,
        confidence=classification_result["parsed"].confidence,
        tokens_used=(
            _tokens_used(classification_result["raw"])
            + _tokens_used(output)
        ),
    )


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
