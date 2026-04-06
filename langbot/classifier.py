import json
from pathlib import Path

from langchain.chat_models import BaseChatModel
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from langbot.models import Classification, RequestType


def classifier(model: BaseChatModel) -> Runnable:
    """Build a classification chain with native structured output or fallback to a parser."""

    # Prefer native structured output when the model/provider supports it.
    structured_model = _structured_output_runnable(model)
    if structured_model is not None:
        prompt = _build_prompt()
        return prompt | structured_model

    # Fallback for models without native schema enforcement.
    parser = PydanticOutputParser(pydantic_object=Classification).with_fallbacks(
        [_unknown_classification()],
        exceptions_to_handle=(OutputParserException,),
    )
    prompt = _build_prompt(parser.get_format_instructions())
    return prompt | model | parser


def _build_prompt(format_instructions: str | None = None) -> ChatPromptTemplate:
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    system_message = (
        "Классифицируй последнее сообщение пользователя в один из классов: "
        "{classes}."
    )
    if format_instructions is not None:
        system_message += "\n\n{format_instructions}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        FewShotChatMessagePromptTemplate(
            examples=_load_examples(),
            example_prompt=example_prompt,
        ),
        ("human", "{input}"),
    ]).partial(classes=", ".join(RequestType))

    if format_instructions is not None:
        prompt = prompt.partial(format_instructions=format_instructions)

    return prompt


def _structured_output_runnable(model: BaseChatModel) -> Runnable | None:
    try:
        return model.with_structured_output(Classification)
    except NotImplementedError:
        return None


def _unknown_classification() -> Runnable:
    return RunnableLambda(
        lambda _: Classification(
            request_type=RequestType.unknown,
            confidence=0.5,
            reasoning="Ошибка парсинга ответа модели",
        )
    )


def _load_examples() -> list[dict]:
    path = Path(__file__).resolve().parent / "prompts" / "classifier_examples.json"
    with path.open(encoding="utf-8") as f:
        examples_by_category = json.load(f)

    mixed_order = [
        ("small_talk", 0),
        ("question", 0),
        ("complaint", 0),
        ("task", 0),
        ("unknown", 0),
        ("task", 1),
        ("question", 1),
        ("small_talk", 1),
        ("unknown", 1),
        ("complaint", 1),
        ("question", 2),
        ("task", 2),
        ("complaint", 2),
        ("small_talk", 2),
        ("unknown", 2),
    ]

    # Keep examples grouped in JSON for editing, but interleave them in the prompt.
    mixed_examples = []
    for category, index in mixed_order:
        example = examples_by_category[category][index]
        mixed_examples.append({
            "input": example["input"],
            # Few-shot AI messages must be plain text, so serialize the JSON object.
            "output": json.dumps(example["output"], ensure_ascii=False),
        })

    return mixed_examples
