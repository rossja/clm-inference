"""Verify official artifact selection, shared caching and encoder matching."""

from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from clm_inference import encoder, heads, server


def test_official_artifacts_use_shared_cache(monkeypatch, tmp_path):
    settings = server.read_settings(Path("config/default.yaml"))
    calls = []

    def download_heads(repo, filename, **kwargs):
        assert not kwargs
        calls.append((repo, filename))
        return str(tmp_path / "heads.pt")

    def download_encoder(repo, **kwargs):
        assert "cache_dir" not in kwargs
        assert "local_dir" not in kwargs
        calls.append(repo)
        return str(tmp_path)

    fake_heads = SimpleNamespace(encoder_model="Qwen/Qwen3-8B")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download_heads)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download_encoder)
    monkeypatch.setattr(heads, "HeadPair", lambda path: fake_heads)
    def make_encoder(path, max_tokens, max_batch_size, memory):
        assert max_tokens == settings.max_tokens
        assert max_batch_size == settings.max_batch_size
        assert memory == settings.gpu_memory_utilization
        return path

    monkeypatch.setattr(encoder, "Encoder", make_encoder)
    engine = server.load_engine(settings)
    assert calls == [
        ("Contrastive-LM/CLM-v0.1-8B", "CLM_v0.1-8B.pt"), "Qwen/Qwen3-8B",
    ]
    assert engine.model == "Contrastive-LM/CLM-v0.1-8B"
    assert engine.encoder == tmp_path
    assert engine.heads is fake_heads


def test_mismatched_heads_fail_before_encoder_download(monkeypatch):
    settings = server.read_settings(Path("config/default.yaml"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda *args: "/unused.pt")
    monkeypatch.setattr(heads, "HeadPair", lambda path: SimpleNamespace(
        encoder_model="incompatible-encoder",
    ))
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="different encoder"):
        server.load_engine(settings)
    assert not calls
