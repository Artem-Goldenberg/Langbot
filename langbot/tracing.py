from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, SecretStr


def default_log_path() -> str:
    return str(Path(".logs") / datetime.now().strftime("%d-%m_%H-%M"))


@dataclass(slots=True)
class _RunState:
    name: str
    started_at: float
    first_token_at: float | None = None


class SessionTraceLogger:
    def __init__(
        self,
        path: str | None,
        *,
        session_info: dict[str, Any],
    ):
        self.path = Path(path) if path else Path(default_log_path())
        self._session_info = session_info
        self._session_started_at = datetime.now()
        self._session_header_written = False
        self._lock = Lock()
        self.log_event("Session Initialized", **session_info)

    def log_event(
        self,
        title: str,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self._ensure_parent_dir()
            parts = []
            if not self._session_header_written:
                parts.append(self._session_header())
                self._session_header_written = True

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            heading = f"[{timestamp}] {title}"
            parts.append(f"{heading}\n")
            if kwargs:
                parts.append(self._indented_json_block(kwargs))
            parts.append("\n")
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write("".join(parts))

    def _session_header(self) -> str:
        session_info = {
            **self._session_info,
            "started_at": self._session_started_at.isoformat(timespec="seconds"),
            "log_path": str(self.path),
        }
        return (
            f"{'=' * 80}\n"
            "Langbot Session Trace\n"
            f"{'=' * 80}\n"
            f"{self._indented_json_block(session_info)}\n"
        )

    def _ensure_parent_dir(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _indented_json_block(self, data: Any) -> str:
        payload = json.dumps(
            _serialize(data),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        return "\n".join(f"  {line}" for line in payload.splitlines()) + "\n"


class LangChainTraceCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        *,
        logger: SessionTraceLogger,
    ):
        self._logger = logger
        self._runs: dict[UUID, _RunState] = {}
        self._chain_names: dict[UUID, str] = {}
        self._chain_parents: dict[UUID, UUID | None] = {}

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self._chain_parents[run_id] = parent_run_id
        self._chain_names[run_id] = _chain_name(
            serialized=serialized,
            kwargs=kwargs,
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        name = _model_event_name(
            parent_run_id=parent_run_id,
            chain_names=self._chain_names,
            chain_parents=self._chain_parents,
            kwargs=kwargs,
        )
        self._runs[run_id] = _RunState(name=name, started_at=time.perf_counter())
        self._logger.log_event(
            f"{name} Model Request",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            tags=tags,
            serialized=serialized,
            messages=messages,
            extra=kwargs or None,
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        name = _model_event_name(
            parent_run_id=parent_run_id,
            chain_names=self._chain_names,
            chain_parents=self._chain_parents,
            kwargs=kwargs,
        )
        self._runs[run_id] = _RunState(name=name, started_at=time.perf_counter())
        self._logger.log_event(
            f"{name} LLM Request",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            tags=tags,
            serialized=serialized,
            prompts=prompts,
            extra=kwargs or None,
        )

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Any:
        run = self._runs.get(run_id)
        if run is None or run.first_token_at is not None:
            return

        run.first_token_at = time.perf_counter()
        self._logger.log_event(
            f"{run.name} First Token",
            run_id=str(run_id),
            ttft_ms=round((run.first_token_at - run.started_at) * 1000, 3),
            token=token,
            chunk=kwargs.get("chunk"),
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        run = self._runs.pop(run_id, None)
        finished_at = time.perf_counter()
        name = run.name if run is not None else _model_event_name(
            parent_run_id=parent_run_id,
            chain_names=self._chain_names,
            chain_parents=self._chain_parents,
            kwargs=kwargs,
        )
        usage = _usage_from_response(response)
        timing = {
            "duration_ms": round(
                (finished_at - run.started_at) * 1000,
                3,
            ) if run is not None else None,
            "ttft_ms": round(
                (run.first_token_at - run.started_at) * 1000,
                3,
            ) if run is not None and run.first_token_at is not None else None,
        }
        record = {
            "name": name,
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags,
            "timing": timing,
            "usage": usage,
        }
        self._logger.log_event(
            f"{name} Model Response",
            **record,
            content_preview=_content_preview(response),
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        run = self._runs.pop(run_id, None)
        finished_at = time.perf_counter()
        name = run.name if run is not None else _model_event_name(
            parent_run_id=parent_run_id,
            chain_names=self._chain_names,
            chain_parents=self._chain_parents,
            kwargs=kwargs,
        )
        self._logger.log_event(
            f"{name} Model Error",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            tags=tags,
            error_type=type(error).__name__,
            error=str(error),
            duration_ms=round(
                (finished_at - run.started_at) * 1000,
                3,
            ) if run is not None else None,
            extra=kwargs or None,
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self._chain_names.pop(run_id, None)
        self._chain_parents.pop(run_id, None)


def _chain_name(*, serialized: dict[str, Any], kwargs: dict[str, Any]) -> str:
    name = kwargs.get("name")
    if name:
        return str(name)
    serialized_name = serialized.get("name")
    if serialized_name:
        return str(serialized_name)
    return "chain"


def _model_event_name(
    *,
    parent_run_id: UUID | None,
    chain_names: dict[UUID, str],
    chain_parents: dict[UUID, UUID | None],
    kwargs: dict[str, Any],
) -> str:
    name = kwargs.get("name")
    if name:
        return str(name)
    ancestor_run_id = parent_run_id
    fallback_name: str | None = None
    while ancestor_run_id is not None:
        ancestor_name = chain_names.get(ancestor_run_id)
        if ancestor_name in {"classifier", "response", "summary", "entity"}:
            return ancestor_name
        if fallback_name is None and ancestor_name not in {None, "chain", "RunnableSequence"}:
            fallback_name = ancestor_name
        ancestor_run_id = chain_parents.get(ancestor_run_id)
    if fallback_name is not None:
        return fallback_name
    return "model"


def _usage_from_response(response: Any) -> dict[str, Any] | None:
    serialized = _serialize(response)

    if isinstance(serialized, dict):
        llm_output = serialized.get("llm_output")
        if isinstance(llm_output, dict) and isinstance(llm_output.get("token_usage"), dict):
            return llm_output["token_usage"]

        generations = serialized.get("generations")
        if isinstance(generations, list):
            for generation_group in generations:
                if not isinstance(generation_group, list):
                    continue
                for generation in generation_group:
                    if not isinstance(generation, dict):
                        continue
                    message = generation.get("message")
                    if not isinstance(message, dict):
                        continue
                    usage = message.get("usage_metadata")
                    if isinstance(usage, dict):
                        return usage

    return None


def _content_preview(response: Any, *, max_length: int = 240) -> str | None:
    serialized = _serialize(response)

    if not isinstance(serialized, dict):
        return None

    generations = serialized.get("generations")
    if not isinstance(generations, list):
        return None

    for generation_group in generations:
        if not isinstance(generation_group, list):
            continue
        for generation in generation_group:
            if not isinstance(generation, dict):
                continue
            text = generation.get("text")
            if isinstance(text, str) and text:
                return _truncate(text, max_length=max_length)
            message = generation.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content:
                    return _truncate(content, max_length=max_length)

    return None


def _truncate(text: str, *, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…"


def _serialize(value: Any, *, _seen: set[int] | None = None) -> Any:
    if _seen is None:
        _seen = set()

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, SecretStr):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, (datetime, Path, UUID)):
        return str(value)

    if isinstance(value, BaseMessage):
        payload = value.model_dump(mode="python")
        payload["message_type"] = value.type
        return _serialize(payload, _seen=_seen)

    if isinstance(value, BaseModel):
        return _serialize(value.model_dump(mode="python"), _seen=_seen)

    if isinstance(value, dict):
        return {
            str(key): _serialize(item, _seen=_seen)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_serialize(item, _seen=_seen) for item in value]

    marker = id(value)
    if marker in _seen:
        return {"repr": repr(value)}
    _seen.add(marker)

    if hasattr(value, "model_dump"):
        try:
            return _serialize(value.model_dump(mode="python"), _seen=_seen)
        except TypeError:
            pass

    if hasattr(value, "dict"):
        try:
            return _serialize(value.dict(), _seen=_seen)
        except TypeError:
            pass

    if hasattr(value, "__dict__"):
        return {
            key: _serialize(item, _seen=_seen)
            for key, item in vars(value).items()
            if not key.startswith("_")
        } or {"repr": repr(value)}

    return {"repr": repr(value)}
