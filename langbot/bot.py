import json
from collections.abc import Iterator
from enum import StrEnum, auto
from pathlib import Path
from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    message_chunk_to_message,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables import RouterRunnable, Runnable, RunnablePassthrough, chain

from langbot.classifier import classifier_chain
from langbot.memory import (
    entity_memory_chain,
    format_entities_json,
    history_fits_context,
    summary_chain,
    trim_history,
)
from langbot.models import *
from langbot.tracing import LangChainTraceCallbackHandler, SessionTraceLogger


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_MAX_CONTEXT_TOKENS = 2_048


class MemoryType(StrEnum):
    buffer = auto()
    summary = auto()


class Bot:
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "{character}\n\n{instructions}{summary}{entities}"),
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
        entity_model: Runnable[Any, AIMessage] | None = None,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        log_path: str | None = None,
    ):
        self.character = character
        self.memory_type = memory_type

        self.main_model = model
        self.classifier_model = classifier_model or model
        self.summary_model = summary_model or model
        self.entity_model = entity_model or model

        self.max_context_tokens = max_context_tokens
        self.summary_chain = summary_chain(self.summary_model).with_config(
            run_name="summary",
        )
        self._summary: str | None = None

        self.entity_chain = entity_memory_chain(self.entity_model).with_config(
            run_name="entity",
        )
        self._entities: dict[str, Any] = {}

        self.trace_logger = SessionTraceLogger(
            log_path,
            session_info={
                "model": _model_name(self.main_model),
                "character": self.character,
                "memory_type": self.memory_type,
            },
        )

        self._history = InMemoryChatMessageHistory()
        self.classifier: Runnable

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
        previous_character = self.character
        self.character = new_character
        self.assemble()
        self.trace_logger.log_event(
            "Character Switched",
            previous_character=previous_character.value,
            new_character=new_character.value,
        )

    def switch_memory(self, new_memory: MemoryType):
        previous_memory = self.memory_type
        self.memory_type = new_memory
        self.trace_logger.log_event(
            "Memory Strategy Switched",
            previous_memory=previous_memory.value,
            new_memory=new_memory.value,
        )

    @property
    def history(self) -> list[BaseMessage]:
        return self._history.messages

    @property
    def entities(self) -> dict[str, Any]:
        return self._entities

    def clear_history(self):
        self._remember_entities(source="clear_history")
        self._history.clear()
        previous_summary = self._summary
        self._set_summary(None)
        self.trace_logger.log_event(
            "History Cleared",
            previous_summary=previous_summary,
        )

    def process(self, input: str) -> AssistanceResponse:
        self.trace_logger.log_event("Started", user_input=input)
        chain_input = self._prepare_chain_input(input)
        result = self.chain.invoke(
            chain_input,
            config=self._trace_config(),
        )
        self._history.add_messages([
            HumanMessage(content=input),
            result["output"],
        ])
        response = _to_assistance_response(result)
        self.trace_logger.log_event(
            "Completed",
            response=response,
            usage_summary={
                "classifier": _usage_metadata(result["classification_result"]["raw"]),
                "response": _usage_metadata(result["output"]),
            },
        )
        return response

    def stream_process(self, input: str) -> Iterator[BotStreamEvent]:
        self.trace_logger.log_event("Started", user_input=input)
        chain_input = self._prepare_chain_input(input)
        classification_result = self.classifier.invoke(
            {"input": input},
            config=self._trace_config(),
        )
        yield _start_event(classification_result)

        streamed_message: AIMessageChunk | None = None

        for chunk in self.stream_response_chain.stream(
            {
                **chain_input,
                "classification_result": classification_result,
            },
            config=self._trace_config(),
        ):
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
            trace_logger=self.trace_logger,
        )
        yield ResponseComplete(response=response)

    def assemble(self):
        self._refresh_main_prompt()

        self.classifier = classifier_chain(self.classifier_model).with_config(
            run_name="classifier",
        )
        router = self._request_type_router()

        self.stream_response_chain = (
            _to_router_input
            | router
        ).with_config(run_name="response")

        self.chain = (
            RunnablePassthrough.assign(classification_result=self.classifier)
            | RunnablePassthrough.assign(output=self.stream_response_chain)
        ).with_config(run_name="bot.process")

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
            self.trace_logger.log_event(
                "History Fits Context",
                message_count=len(history),
            )
            return history

        self._remember_entities(source="history_reduction")

        match self.memory_type:
            case MemoryType.buffer:
                trimmed_history = trim_history(
                    history,
                    max_tokens=self.max_context_tokens,
                    model=self.main_model,
                )
                self.trace_logger.log_event(
                    "History Trimmed",
                    memory_type=self.memory_type.value,
                    message_count_before=len(history),
                    message_count_after=len(trimmed_history),
                )
                return trimmed_history
            case MemoryType.summary:
                self.trace_logger.log_event(
                    "Summary Memory Triggered",
                    message_count=len(history),
                )
                self._set_summary(
                    self.summary_chain.invoke(
                        {"history": history},
                        config=self._trace_config(),
                    ),
                )
                return []

    def _set_summary(self, summary: str | None):
        previous_summary = self._summary
        self._summary = summary
        self.assemble()
        if previous_summary != summary:
            self.trace_logger.log_event(
                "Summary Updated",
                previous_summary=previous_summary,
                new_summary=summary,
            )

    def _remember_entities(self, *, source: str):
        if not self.history:
            return

        previous_entities = self._entities
        self.trace_logger.log_event(
            title="Entity Memory Triggered",
            source=source,
            message_count=len(self.history),
            known_entity_count=len(previous_entities),
        )

        try:
            extracted = self.entity_chain.invoke({
                "known_entities": self._entities,
                "history": self.history,
            }, config=self._trace_config())
        except OutputParserException:
            self.trace_logger.log_event(
                title="Entity Memory Parse Failed",
                source=source,
            )
            return

        if not isinstance(extracted, dict):
            self.trace_logger.log_event(
                title="Entity Memory Ignored Non-Object Output",
                source=source,
                output_type=type(extracted).__name__,
            )
            return

        self._entities = _merge_json_objects(self._entities, extracted)
        self.assemble()
        self.trace_logger.log_event(
            title="Entity Memory Updated",
            source=source,
            previous_entities=previous_entities,
            extracted_entities=extracted,
            merged_entities=self._entities,
        )

    def _refresh_main_prompt(self):
        self.main_prompt = self.prompt_template.partial(
            character=self._character_prompts[self.character],
            summary=_summary_suffix(self._summary),
            entities=_entities_suffix(self._entities),
        )

    def _trace_config(self) -> RunnableConfig:
        return {
            "callbacks": [LangChainTraceCallbackHandler(logger=self.trace_logger)],
        }


def _summary_suffix(summary: str | None) -> str:
    if summary is None:
        return ""
    return f"\n\nКраткая сводка предыдущего диалога:\n{summary}"


def _entities_suffix(entities: dict[str, Any]) -> str:
    if not entities:
        return ""
    return (
        "\n\nИзвестные данные о пользователе:\n"
        f"{format_entities_json(entities)}"
    )


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
    trace_logger: SessionTraceLogger,
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
    trace_logger.log_event(
        "Completed",
        response=response,
        usage_summary={
            "classifier": _usage_metadata(classification_result["raw"]),
            "response": _usage_metadata(streamed_message),
        },
    )
    return response


def _final_message(chunk: AIMessageChunk | None) -> AIMessage:
    if chunk is None:
        return AIMessage(content="")
    message = message_chunk_to_message(chunk)
    if not isinstance(message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(message).__name__}")
    return message


def _merge_json_objects(
    current: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(current)
    for key, value in updates.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_json_objects(existing, value)
            continue
        merged[key] = value
    return merged


def _model_name(model: Runnable[Any, AIMessage]) -> str:
    bound = getattr(model, "bound", None)
    if bound is not None:
        return _model_name(bound)

    model_name = getattr(model, "model_name", None) or getattr(model, "model", None)
    if model_name is not None:
        return str(model_name)
    return type(model).__name__
