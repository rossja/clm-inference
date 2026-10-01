"""Generic CLM request types, text formatting and answer distributions.

Matches the upstream CLM System One contract:
https://github.com/Contrastive-LM/CLM/blob/main/src/clm/schema.py
"""

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


class QuestionBase(BaseModel):
    """Caller-provided instructions, independent of application domain."""

    model_config = ConfigDict(extra="forbid")
    instructions: str = Field(min_length=1)


class Choice(QuestionBase):
    """Select a label from caller-supplied descriptions."""

    type: Literal["choice"]
    criteria: dict[str, JsonValue] = Field(min_length=1)


class Noul(QuestionBase):
    """Estimate the probability that a caller-supplied statement is true."""

    type: Literal["noul"]
    criteria: dict[str, JsonValue] | None = None

    @field_validator("criteria")
    @classmethod
    def validate_keys(cls, value):
        """Only true and false have meaning for a binary statement."""
        if value is not None and set(value) - {"true", "false"}:
            raise ValueError("noul criteria keys must be 'true' or 'false'")
        return value


class Score(QuestionBase):
    """Score a state against an ordered rubric supplied by the caller."""

    type: Literal["score"]
    criteria: list[JsonValue] = Field(min_length=2)


Question = Annotated[Choice | Noul | Score, Field(discriminator="type")]
State = str | dict[str, JsonValue] | list[JsonValue]


class DecisionRequest(BaseModel):
    """Ask one or more typed questions about a state."""

    model_config = ConfigDict(extra="forbid")
    state: State
    questions: dict[str, Question] = Field(min_length=1)
    model: str | None = None
    temperature: float = Field(default=1.0, gt=0, le=100, allow_inf_nan=False)


class RankRequest(BaseModel):
    """Rank caller-supplied answers against a context and question."""

    model_config = ConfigDict(extra="forbid")
    context: State
    question: str = Field(min_length=1)
    answers: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    model: str | None = None
    temperature: float = Field(default=1.0, gt=0, le=100, allow_inf_nan=False)


class ChoiceAnswer(BaseModel):
    """Winning label and distribution over all supplied labels."""

    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class NoulAnswer(BaseModel):
    """Probability the statement is true; false has probability 1 - noul."""

    type: Literal["noul"] = "noul"
    noul: float


class ScoreAnswer(BaseModel):
    """Expected rubric index and distribution over the ordered levels."""

    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[str, str]
    probabilities: dict[str, float]


Answer = Annotated[
    ChoiceAnswer | NoulAnswer | ScoreAnswer, Field(discriminator="type")
]


class DecisionUsage(BaseModel):
    """Questions answered and encoder tokens used after truncation."""

    billing_units: int
    input_tokens: int
    output_tokens: Literal[0] = 0


class DecisionResponse(BaseModel):
    """Typed answers keyed by the caller's question identifiers."""

    model: str
    answers: dict[str, Answer]
    usage: DecisionUsage


class RankedCandidate(BaseModel):
    """One candidate and its probability, ordered by descending probability."""

    rank: int
    candidate: str
    prob: float


class RankResponse(BaseModel):
    """Ranked candidates from the same CLM scoring primitive."""

    model: str
    ranked: list[RankedCandidate]


def to_text(value: JsonValue, indent: int = 0) -> str:
    """Render structured input as prose using upstream CLM's formatting."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    padding = " " * indent
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if isinstance(item, (dict, list)) and item:
                parts.append(f"{padding}{key}:\n{to_text(item, indent + 2)}")
            else:
                parts.append(f"{padding}{key}: {to_text(item)}")
        return ("\n\n" if indent == 0 else "\n").join(parts)
    if isinstance(value, list):
        return "\n".join(
            f"{padding}-\n{to_text(item, indent + 2)}"
            if isinstance(item, (dict, list)) and item
            else f"{padding}- {to_text(item)}"
            for item in value
        )
    return str(value)


def state_text(state: State, instructions: str) -> str:
    """Place the question after the context, as in CLM training."""
    context, question = to_text(state).strip(), instructions.strip()
    return f"{context}\n\n{question}" if context and question else (
        context or question
    )


def candidates(question: Question) -> tuple[list[str], list[str]]:
    """Return option identifiers and the texts seen by the action head."""
    if isinstance(question, Choice):
        keys = list(question.criteria)
        return keys, [
            key if question.criteria[key] in (None, "")
            else to_text(question.criteria[key]) for key in keys
        ]
    if isinstance(question, Score):
        return (
            [str(index) for index in range(len(question.criteria))],
            [to_text(level) for level in question.criteria],
        )
    descriptions = question.criteria or {}
    keys = ["false", "true"]
    texts = []
    for key, prefix in zip(keys, ["No. This is false", "Yes. This is true"]):
        description = descriptions.get(key)
        if description in (None, ""):
            description = f"{prefix}: {question.instructions.strip()}"
        texts.append(f"{key}: {to_text(description)}")
    return keys, texts


def answer_from_logits(
    question: Question, keys: list[str], logits: list[float]
) -> Answer:
    """Convert scaled similarities into upstream CLM's typed answer format."""
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    distribution = dict(zip(keys, probabilities))
    if isinstance(question, Noul):
        return NoulAnswer(noul=distribution["true"])
    winner = max(range(len(keys)), key=probabilities.__getitem__)
    confidence = 1.0 if len(keys) == 1 else (
        probabilities[winner]
        - (1.0 - probabilities[winner]) / (len(keys) - 1)
    )
    confidence = max(0.0, min(1.0, confidence))
    if isinstance(question, Choice):
        return ChoiceAnswer(
            choice=keys[winner], confidence=confidence,
            probabilities=distribution,
        )
    return ScoreAnswer(
        score=sum(index * prob for index, prob in enumerate(probabilities)),
        confidence=confidence,
        legend={key: to_text(level)
                for key, level in zip(keys, question.criteria)},
        probabilities=distribution,
    )
