from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages import trim_messages
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.runnables.base import RunnableBindingBase
from langchain_core.runnables.fallbacks import RunnableWithFallbacks


def summary_chain(model: Runnable[Any, AIMessage]) -> Runnable:
    """Build a chain that compresses chat history into a short working summary."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Ты сжимаешь историю диалога в короткую рабочую сводку для ассистента. "
            "Сохраняй факты, задачи, предпочтения пользователя, обещания и открытые "
            "вопросы. Пиши кратко, без вводных фраз и без выдуманных деталей.",
        ),
        MessagesPlaceholder("history"),
    ])
    return prompt | model | StrOutputParser()


def entity_memory_chain(model: Runnable[Any, AIMessage]) -> Runnable:
    """Build a chain that extracts durable user facts as a JSON object."""
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Ты извлекаешь долговременные факты о пользователе из истории диалога. "
            "Сохраняй имена, города, предпочтения, биографические сведения, важные "
            "устойчивые настройки и прочие полезные факты, которые помогут ассистенту "
            "помнить пользователя после очистки истории. Не добавляй временные детали, "
            "догадки и факты, которых нет в истории. Верни только один JSON-объект без "
            "объяснений. Верхний уровень JSON — это карта сущностей: каждый ключ — имя "
            "сущности, значение — любой валидный JSON. Используй уже известные сущности "
            "как базу и аккуратно обновляй их, если история даёт более точные данные.\n"
            "{format_instructions}\n\n"
            "Уже известные сущности:\n{known_entities}",
        ),
        MessagesPlaceholder("history"),
    ])
    return (
        RunnablePassthrough.assign(
            known_entities=lambda data: format_entities_json(data["known_entities"])
        )
        | prompt.partial(
            format_instructions=parser.get_format_instructions(),
        )
        | model
        | parser
    )


def format_entities_json(entities: dict[str, Any]) -> str:
    """Render the entity store as stable, human-readable JSON."""
    return json.dumps(entities, ensure_ascii=False, indent=2, sort_keys=True)


def history_fits_context(
    history: list[BaseMessage],
    *,
    model: Runnable[Any, AIMessage],
    max_tokens: int,
) -> bool:
    """Return whether the current history fits inside the target token budget."""
    return _chat_model(model).get_num_tokens_from_messages(history) <= max_tokens


def trim_history(
    history: list[BaseMessage],
    *,
    model: Runnable[Any, AIMessage],
    max_tokens: int,
) -> list[BaseMessage]:
    """Trim history to the token budget while keeping a valid conversation shape."""
    return trim_messages(
        history,
        max_tokens=max_tokens,
        token_counter=_chat_model(model),
        start_on="human",
        include_system=True,
    )


def _chat_model(model: Runnable[Any, AIMessage] | BaseLanguageModel) -> BaseLanguageModel:
    """Unwrap runnable bindings until a token-counting chat model is reached."""
    if isinstance(model, RunnableWithFallbacks):
        return _chat_model(model.runnable)

    if isinstance(model, RunnableBindingBase):
        return _chat_model(model.bound)

    if isinstance(model, BaseLanguageModel):
        return model

    raise TypeError(f"Unsupported model type for token counting: {type(model).__name__}")
