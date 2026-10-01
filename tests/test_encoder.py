"""Check last-token selection, truncation and normalization using real MLX."""

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from clm_inference import encoder


def test_pooling_contract(monkeypatch):
    calls = []

    class Tokenizer:
        """Use known token IDs to make truncation observable."""

        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return {"long": [1, 2, 3], "short": [4], "": [], " ": [5]}[text]

    class Model:
        """Return distinct hidden states for each token."""

        def __call__(self, inputs, *, cache):
            assert cache is None
            calls.append(inputs.tolist())
            values = inputs.astype(mx.float32)
            return mx.stack([values, mx.ones_like(values)], axis=-1)

    model = SimpleNamespace(model=Model(), parameters=lambda: [])
    monkeypatch.setattr(encoder, "load", lambda *args, **kwargs: (model, Tokenizer()))
    instance = encoder.Encoder(Path("unused"))
    vectors, tokens = instance.embed(["long", "short", ""], max_tokens=2)
    assert calls == [[[1, 2]], [[4]], [[5]]]
    assert tokens == 4
    for vector, last_token in zip(vectors, [2, 4, 5]):
        norm = (last_token ** 2 + 1) ** 0.5
        assert vector == pytest.approx([last_token / norm, 1 / norm])
