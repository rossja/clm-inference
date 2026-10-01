"""Check generic question handling, scoring and upstream wire semantics."""

import math
from types import SimpleNamespace

import pytest

from clm_inference.engine import Engine
from clm_inference.schema import (
    Choice, DecisionRequest, Noul, RankRequest, Score, answer_from_logits,
    candidates, state_text, to_text,
)


def test_multi_question_scoring_and_temperature():
    calls = []

    def embed(texts, max_tokens):
        assert max_tokens == 20
        calls.append(texts)
        return [[float(index)] for index in range(len(texts))], len(texts)

    def logits(states, actions):
        assert len(states) == 3
        assert len(actions) == 7
        return [
            [0, 2, 99, 99, 99, 99, 99],
            [99, 99, 0, 2, 99, 99, 99],
            [99, 99, 99, 99, 0, 2, 4],
        ]

    engine = Engine(SimpleNamespace(embed=embed), SimpleNamespace(logits=logits),
                    "test", 20)
    request = DecisionRequest(state={"detail": "context"}, temperature=2,
                              questions={
        "category": Choice(type="choice", instructions="Select",
                           criteria={"a": "First", "b": "Second"}),
        "statement": Noul(type="noul", instructions="It is correct"),
        "level": Score(type="score", instructions="Assess",
                       criteria=["Low", "Medium", "High"]),
    })
    response = engine.answer(request)
    assert calls[0] == [
        "detail: context\n\nSelect", "detail: context\n\nIt is correct",
        "detail: context\n\nAssess",
    ]
    assert calls[1] == [
        "First", "Second", "false: No. This is false: It is correct",
        "true: Yes. This is true: It is correct", "Low", "Medium", "High",
    ]
    probability = math.e / (1 + math.e)
    assert response.answers["category"].choice == "b"
    assert response.answers["category"].probabilities["b"] == pytest.approx(
        probability
    )
    assert response.answers["statement"].noul == pytest.approx(probability)
    assert response.answers["level"].score == pytest.approx(
        (math.e + 2 * math.e ** 2) / (1 + math.e + math.e ** 2)
    )
    assert response.usage.input_tokens == 10
    assert response.usage.billing_units == 3
    assert response.usage.output_tokens == 0


def test_upstream_formatting_and_candidates():
    assert to_text({"subject": "hello", "items": ["a", {"flag": True}]}) == (
        "subject: hello\n\nitems:\n  - a\n  -\n    flag: true"
    )
    assert state_text(" context ", " question ") == "context\n\nquestion"
    assert candidates(Choice(type="choice", instructions="Pick", criteria={
        "a": "", "b": None, "c": {"description": "structured"},
    })) == (["a", "b", "c"], ["a", "b", "description: structured"])
    assert candidates(Noul(type="noul", instructions="True?", criteria={
        "false": "No", "true": "Yes",
    })) == (["false", "true"], ["false: No", "true: Yes"])


def test_stable_softmax_confidence_and_score():
    question = Choice(type="choice", instructions="Pick", criteria={
        "a": "A", "b": "B", "c": "C",
    })
    answer = answer_from_logits(question, ["a", "b", "c"], [1000] * 3)
    assert answer.choice == "a"
    assert answer.confidence == pytest.approx(0)
    assert sum(answer.probabilities.values()) == pytest.approx(1)
    single = Choice(type="choice", instructions="Pick", criteria={"a": "A"})
    assert answer_from_logits(single, ["a"], [1000]).confidence == 1
    score = Score(type="score", instructions="Rate", criteria=["Low", "High"])
    result = answer_from_logits(score, ["0", "1"], [0, 0])
    assert result.score == 0.5
    assert result.legend == {"0": "Low", "1": "High"}


def test_ranking_preserves_duplicate_candidates():
    engine = Engine(None, None, "test", 20)
    seen = []

    def answer(request):
        seen.append(request)
        return SimpleNamespace(answers={"rank": SimpleNamespace(
            probabilities={"0": 0.1, "1": 0.6, "2": 0.3},
        )})

    engine.answer = answer
    response = engine.rank(RankRequest(
        context="context", question="question", answers=["same", "best", "same"],
        temperature=2,
    ))
    assert seen[0].questions["rank"].criteria == {
        "0": "same", "1": "best", "2": "same",
    }
    assert seen[0].temperature == 2
    assert [item.candidate for item in response.ranked] == ["best", "same", "same"]
    assert [item.rank for item in response.ranked] == [1, 2, 3]
