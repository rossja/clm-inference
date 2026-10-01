"""Check token boundaries, batching and lifecycle at the vLLM adapter."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from clm_inference.encoder import Encoder


@pytest.mark.parametrize("environment", [None, "1"])
def test_encoder_contract(monkeypatch, environment):
    calls = []
    if environment is None:
        monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    else:
        monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", environment)

    class Tokenizer:
        """Use known token IDs to make truncation observable."""

        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return {"long": [1, 2, 3], "short": [4], "": [], " ": [5]}[text]

    class Model:
        """Record initialization and token prompts without loading weights."""

        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(
                shutdown=self.shutdown,
            ))

        def shutdown(self):
            torch.accelerator.empty_host_cache()
            calls.append("closed")

        def get_tokenizer(self):
            return Tokenizer()

        def embed(self, prompts, *, use_tqdm):
            assert not use_tqdm
            calls.append(prompts)
            return [SimpleNamespace(outputs=SimpleNamespace(
                embedding=[float(index)],
            )) for index in range(len(prompts))]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Model))
    instance = Encoder(Path("model"), 10, 5, 0.5)
    vectors, tokens = instance.embed(["long", "short", ""], max_tokens=2)
    assert calls[0] == {
        "model": "model", "runner": "pooling", "dtype": "bfloat16",
        "max_model_len": 10, "max_num_seqs": 5,
        "gpu_memory_utilization": 0.5, "enforce_eager": True,
        "enable_prefix_caching": False,
    }
    assert calls[1] == [
        {"prompt_token_ids": [1, 2]}, {"prompt_token_ids": [4]},
        {"prompt_token_ids": [5]},
    ]
    assert vectors == [[0.0], [1.0], [2.0]]
    assert tokens == 4
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == (environment or "0")
    original = torch.accelerator.empty_host_cache
    monkeypatch.setattr(torch.mps, "empty_cache",
                        lambda: calls.append("mps-cache-cleared"))
    instance.close()
    assert calls[-2] == "mps-cache-cleared"
    assert calls[-1] == "closed"
    assert torch.accelerator.empty_host_cache is original


def test_cleanup_restores_torch_after_error():
    """A failing teardown must not leave PyTorch's API replaced globally."""
    def fail():
        raise RuntimeError("shutdown failed")

    instance = Encoder.__new__(Encoder)
    instance.model = SimpleNamespace(llm_engine=SimpleNamespace(
        engine_core=SimpleNamespace(shutdown=fail),
    ))
    original = torch.accelerator.empty_host_cache
    with pytest.raises(RuntimeError, match="shutdown failed"):
        instance.close()
    assert torch.accelerator.empty_host_cache is original
