"""Qwen3 last-token embeddings through vLLM's Apple Silicon Metal plugin."""

import os
from pathlib import Path
from unittest.mock import patch

import torch


class Encoder:
    """Use vLLM Metal scheduling, pooling and normalization for the encoder."""

    def __init__(
        self, model_path: Path, max_tokens: int, max_batch_size: int,
        gpu_memory_utilization: float,
    ):
        # Metal's documented single-process mode keeps model ownership on the
        # application's inference thread. Respect an explicit environment value.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM  # pylint: disable=import-outside-toplevel

        self.model = LLM(
            model=str(model_path), runner="pooling", dtype="bfloat16",
            max_model_len=max_tokens, max_num_seqs=max_batch_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True, enable_prefix_caching=False,
        )
        self.tokenizer = self.model.get_tokenizer()

    def embed(
        self, texts: list[str], max_tokens: int
    ) -> tuple[list[list[float]], int]:
        """Submit bounded token inputs together, preserving the input order."""
        prompts = []
        token_count = 0
        for text in texts:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                token_ids = self.tokenizer.encode(" ", add_special_tokens=False)
            token_ids = token_ids[:max_tokens]
            token_count += len(token_ids)
            prompts.append({"prompt_token_ids": token_ids})
        outputs = self.model.embed(prompts, use_tqdm=False)
        return [output.outputs.embedding for output in outputs], token_count

    def close(self) -> None:
        """Release vLLM's engine resources on the model's owning thread."""
        # PyTorch 2.13's generic host-cache cleanup segfaults on MPS. vLLM calls
        # it during teardown. Use MPS's supported cleanup only for that scope;
        # restore the original function even if another shutdown step fails.
        with patch.object(torch.accelerator, "empty_host_cache",
                          torch.mps.empty_cache):
            self.model.llm_engine.engine_core.shutdown()
