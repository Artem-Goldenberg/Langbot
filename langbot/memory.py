from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages import trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.base import RunnableBindingBase


def summary_chain(model: Runnable[Any, AIMessage]) -> Runnable:
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


def history_fits_context(
    history: list[BaseMessage],
    *,
    model: Runnable[Any, AIMessage],
    max_tokens: int,
) -> bool:
    return _chat_model(model).get_num_tokens_from_messages(history) <= max_tokens


def trim_history(
    history: list[BaseMessage],
    *,
    model: Runnable[Any, AIMessage],
    max_tokens: int,
) -> list[BaseMessage]:
    return trim_messages(
        history,
        max_tokens=max_tokens,
        token_counter=_chat_model(model),
        start_on="human",
        include_system=True,
    )


def _chat_model(model: Runnable[Any, AIMessage] | BaseLanguageModel) -> BaseLanguageModel:
    if isinstance(model, RunnableBindingBase):
        return _chat_model(model.bound)

    if isinstance(model, BaseLanguageModel):
        return model

    raise TypeError(f"Unsupported model type for token counting: {type(model).__name__}")
