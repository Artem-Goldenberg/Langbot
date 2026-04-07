import json
from enum import Enum
from pathlib import Path
from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RouterRunnable,
    Runnable,
    RunnablePassthrough,
    chain,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from langbot.models import AssistanceResponse, Character, RequestType
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
        self.main_prompt = self.prompt_template

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
        ...

    def switch_memory(self, new_memory: MemoryType):
        self.memory = new_memory
        ...

    @property
    def history(self) -> list[BaseMessage]:
        return self._history.messages

    def clear_history(self):
        self._history.clear()
        ...

    def process(self, input: str):
        return self.chain.invoke({"input": input})

    def assemble(self):
        router = self._request_type_router()

        classifier = classifier_chain(self.classifier_model)

        response_chain = (
            RunnablePassthrough.assign(classification_result=classifier)
            | RunnablePassthrough.assign(
                output = _to_router_input | router
            )
        )

        chain_with_history = RunnableWithMessageHistory(
            response_chain,
            lambda: self._history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="output",
        )

        self.chain = chain_with_history | _to_assistance_response

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
    output = data["output"]
    return AssistanceResponse.model_validate({
        "content": output.content,
        "request_type": data["classification_result"]["parsed"].request_type,
        "confidence": data["classification_result"]["parsed"].confidence,
        "tokens_used": (
            _tokens_used(data["classification_result"]["raw"])
            + _tokens_used(output)
        ),
    })


def _load_json(filename: str) -> dict[str, str]:
    with (_PROMPTS_DIR / filename).open(encoding="utf-8") as f:
        return json.load(f)


def _tokens_used(message: AIMessage) -> int:
    if message.usage_metadata is None:
        return 0
    return message.usage_metadata.get("total_tokens", 0)
