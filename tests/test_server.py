"""API contract checks without loading model weights."""

import json
import os
from pathlib import Path
import threading

from fastapi.testclient import TestClient
import pytest

from clm_inference import server


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

    def load_encoder(model):
        assert model == "test-model"
        calls.append(threading.get_ident())
        return FakeEncoder()

    monkeypatch.setattr(server, "load_encoder", load_encoder)
    settings = server.Settings(
        model="test-model", host="127.0.0.1", port=8092,
        max_tokens=10, max_batch_size=2,
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
    ({"input": ["a", "b", "c"]}, 400),
    ({"input": "a", "model": "wrong"}, 400),
    ({"input": "a", "truncate_prompt_tokens": 11}, 400),
    ({"input": "a", "truncate_prompt_tokens": 0}, 422),
    ({"input": "a", "encoding_format": "base64"}, 422),
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
        "max_tokens: 10\nmax_batch_size: 2\n", encoding="utf-8",
    )
    assert server.read_settings(config).model == "test-model"
    assert os.environ["HF_HOME"] == "file-cache"


def test_openapi_schema_is_current():
    root = Path(__file__).resolve().parents[1]
    settings = server.read_settings(root / "config/default.yaml")
    schema = json.loads((root / "swagger.json").read_text(encoding="utf-8"))
    assert server.create_app(settings).openapi() == schema
