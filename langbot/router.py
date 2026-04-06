import json
from pathlib import Path

from langchain.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RouterRunnable, Runnable
from langchain_core.prompts import ChatPromptTemplate

from langbot.models import Character, RequestType


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def routed_bot_chain(model: BaseChatModel, character: Character) -> Runnable:
    """Build a router chain that dispatches by request-type key.

    The resulting runnable expects input in RouterRunnable format:
    `{"key": str(RequestType.*), "input": {"input": "..."}}`.
    Each key is routed to its own prompt path with shared character style.
    """
    bot_prompt = ChatPromptTemplate.from_messages([
        ("system", "{character}\n\n{instructions}"),
        ("human", "{input}")
    ])

    def chain_for(instructions: str) -> Runnable:
        prompt = bot_prompt.partial(
            character=character_prompts[character],
            instructions=instructions
        )
        return prompt | model | StrOutputParser()

    # RouterRunnable chooses a different chain depending on the incoming key.
    routes = {str(type): chain_for(prompt) for type, prompt in prompts.items()}

    return RouterRunnable(routes)


def _load_json(filename: str) -> dict[str, str]:
    with (_PROMPTS_DIR / filename).open(encoding="utf-8") as f:
        return json.load(f)


prompts = {
    request_type: prompt
    for request_type, prompt in (
        (RequestType(key), value)
        for key, value in _load_json("request_prompts.json").items()
    )
}


character_prompts = _load_json("character_prompts.json")
character_prompts = {
    Character(key): value for key, value in character_prompts.items()
}
