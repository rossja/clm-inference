"""API contract checks without loading model weights."""

import base64
import json
import os
from pathlib import Path
import struct
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from clm_inference import server
from clm_inference.schema import (
    ChoiceAnswer, DecisionResponse, DecisionUsage, RankedCandidate, RankResponse,
)


@pytest.fixture
def client(monkeypatch):
    """Replace the encoder while retaining the real application lifespan."""
    calls = []

    class FakeEncoder:
        """Record calls and verify inference stays on the loading thread."""

        def embed(self, texts, max_tokens):
            assert threading.get_ident() == calls[0]
            calls.append((texts, max_tokens))
            return [[float(index)] for index in range(len(texts))], len(texts)

    def answer(request):
        assert threading.get_ident() == calls[0]
        calls.append(request)
        return DecisionResponse(
            model="test-model",
            answers={key: ChoiceAnswer(choice="a", confidence=0.5,
                                      probabilities={"a": 0.75, "b": 0.25})
                     for key in request.questions},
            usage=DecisionUsage(billing_units=len(request.questions),
                                input_tokens=3),
        )

    def rank(request):
        assert threading.get_ident() == calls[0]
        calls.append(request)
        return RankResponse(
            model="test-model",
            ranked=[RankedCandidate(rank=1, candidate=request.answers[0],
                                    prob=1.0)],
        )

    def load_engine(settings):
        assert settings.model == "test-model"
        calls.append(threading.get_ident())
        return SimpleNamespace(encoder=FakeEncoder(), answer=answer, rank=rank)

    monkeypatch.setattr(server, "load_engine", load_engine)
    settings = server.Settings(
        model="test-model", host="127.0.0.1", port=8092,
        heads_config="heads/config.json", heads_weights="heads/test.safetensors",
        max_tokens=10, max_batch_size=5,
    )
    with TestClient(server.create_app(settings)) as test_client:
        yield test_client, calls


def test_embeddings_preserve_order_and_usage(client):
    test_client, calls = client
    response = test_client.post("/v1/embeddings", json={
        "input": ["first", "second"], "truncate_prompt_tokens": 3,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "test-model"
    assert body["data"] == [
        {"object": "embedding", "index": 0, "embedding": [0.0]},
        {"object": "embedding", "index": 1, "embedding": [1.0]},
    ]
    assert body["usage"] == {"prompt_tokens": 2, "total_tokens": 2}
    assert calls[1] == (["first", "second"], 3)


def test_string_and_health(client):
    test_client, calls = client
    assert test_client.get("/health").json() == {
        "status": "ok", "model": "test-model",
    }
    assert test_client.post("/v1/embeddings", json={"input": "one"}).status_code == 200
    assert calls[1] == (["one"], 10)


@pytest.mark.parametrize("body,status", [
    ({"input": []}, 400),
    ({"input": ["a", "b", "c", "d", "e", "f"]}, 400),
    ({"input": "a", "model": "wrong"}, 400),
    ({"input": "a", "truncate_prompt_tokens": 11}, 400),
    ({"input": "a", "truncate_prompt_tokens": 0}, 422),
    ({"input": "a", "encoding_format": "unknown"}, 422),
    ({"input": [1, 2]}, 422),
])
def test_reject_invalid_inputs(client, body, status):
    test_client, calls = client
    assert test_client.post("/v1/embeddings", json=body).status_code == status
    assert len(calls) == 1


def test_dotenv_overrides_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HOME", "environment-cache")
    (tmp_path / ".env").write_text("HF_HOME=file-cache\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "model: test-model\nhost: 127.0.0.1\nport: 8092\n"
        "heads_config: heads/config.json\nheads_weights: heads/test.safetensors\n"
        "max_tokens: 10\nmax_batch_size: 2\n", encoding="utf-8",
    )
    assert server.read_settings(config).model == "test-model"
    assert os.environ["HF_HOME"] == "file-cache"


def test_base64_embeddings_for_upstream_clients(client):
    test_client, _ = client
    response = test_client.post("/v1/embeddings", json={
        "input": ["one", "two"], "encoding_format": "base64",
    })
    assert response.status_code == 200
    decoded = [
        struct.unpack("<f", base64.b64decode(item["embedding"]))[0]
        for item in response.json()["data"]
    ]
    assert decoded == [0.0, 1.0]


def test_openapi_schema_is_current():
    root = Path(__file__).resolve().parents[1]
    settings = server.read_settings(root / "config/default.yaml")
    schema = json.loads((root / "swagger.json").read_text(encoding="utf-8"))
    assert server.create_app(settings).openapi() == schema


def test_decision_and_rank_endpoints(client):
    test_client, calls = client
    decision = test_client.post("/v1/systemone", json={
        "state": "any domain", "questions": {
            "q": {"type": "choice", "instructions": "Choose an option",
                  "criteria": {"a": "First", "b": "Second"}},
        },
    })
    assert decision.status_code == 200
    assert decision.json()["answers"]["q"]["choice"] == "a"
    ranked = test_client.post("/v1/rank", json={
        "context": "any domain", "question": "Pick a candidate",
        "answers": ["first", "second"],
    })
    assert ranked.status_code == 200
    assert ranked.json()["ranked"][0]["candidate"] == "first"
    assert len(calls) == 3
    assert test_client.get("/v1/models").json() == {
        "models": [{"name": "test-model"}],
    }


@pytest.mark.parametrize("endpoint,body,status", [
    ("systemone", {"state": "a", "questions": {}}, 422),
    ("systemone", {"state": "a", "questions": {
        "q": {"type": "choice", "instructions": "Pick", "criteria": {}}}}, 422),
    ("systemone", {"state": "a", "questions": {
        "q": {"type": "unknown", "instructions": "Pick"}}}, 422),
    ("systemone", {"state": "a", "questions": {
        "q": {"type": "score", "instructions": "Pick", "criteria": ["one"]}}},
     422),
    ("systemone", {"state": "a", "questions": {
        "q": {"type": "noul", "instructions": "Pick",
              "criteria": {"malicious": "wrong key"}}}}, 422),
    ("systemone", {"state": "a", "questions": {
        "q": {"type": "noul", "instructions": "Pick"}}, "temperature": 0}, 422),
    ("systemone", {"state": "a", "questions": {
        "q": {"type": "noul", "instructions": "Pick"}}, "model": "wrong"}, 400),
    ("systemone", {"state": "a", "questions": {
        "q": {"type": "choice", "instructions": "Pick",
              "criteria": {str(i): str(i) for i in range(5)}}}}, 400),
    ("rank", {"context": "a", "question": "Pick", "answers": []}, 422),
    ("rank", {"context": "a", "question": "Pick", "answers": [""]}, 422),
    ("rank", {"context": "a", "question": "Pick", "answers": ["a"] * 5}, 400),
    ("rank", {"context": "a", "question": "Pick", "answers": ["a"],
              "model": "wrong"}, 400),
])
def test_reject_invalid_decisions(client, endpoint, body, status):
    test_client, calls = client
    assert test_client.post(f"/v1/{endpoint}", json=body).status_code == status
    assert len(calls) == 1
