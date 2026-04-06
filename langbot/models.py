from inspect import cleandoc
from enum import StrEnum, auto
from pydantic import BaseModel, ConfigDict, Field


class RequestType(StrEnum):
    question = auto()
    task = auto()
    small_talk = auto()
    complaint = auto()
    unknown = auto()


class Character(StrEnum):
    friendly = auto()
    professional = auto()
    sarcastic = auto()
    pirate = auto()


class Classification(BaseModel):
    request_type: RequestType = Field(
        description=cleandoc("""
            Тип запроса:
            question — вопрос, требующий информации («Что такое Python?», «Как работает GIL?»)
            task — просьба что-то сделать («Напиши стих», «Расскажи анекдот»)
            small_talk — приветствие и болтовня («Привет!», «Как дела?»)
            complaint — жалоба, недовольство («Это ужасно работает!», «Почему так долго?»)
            unknown — бессмыслица или нераспознанный запрос («asdfghjkl»)
        """)
    )

    confidence: float = Field(
        ge=0, le=1, description="насколько ты уверен в классификации"
    )

    reasoning: str = Field(description="краткое обоснование")


class AssistanceResponse(BaseModel):
    content: str = Field(min_length=1, description="текст ответа")

    request_type: RequestType = Field(description="какой тип запроса был определён")

    confidence: float = Field(ge=0, le=1, description="уверенность")

    tokens_used: int = Field(
        ge=0, description="приблизительное число токенов (для мониторинга)"
    )
