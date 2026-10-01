"""Last-token pooling for the Qwen3 encoder used by CLM."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load


class Encoder:
    """Load one encoder and produce L2-normalized, float32 embeddings."""

    def __init__(self, model_path: Path):
        self.model, self.tokenizer = load(str(model_path), lazy=True)
        # Discard the unused generation head before evaluation.
        self.model.lm_head = nn.Identity()
        mx.eval(self.model.parameters())

    def embed(
        self, texts: list[str], max_tokens: int
    ) -> tuple[list[list[float]], int]:
        """Embed individually to avoid padding and large batch allocations."""
        embeddings = []
        token_count = 0
        for text in texts:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                token_ids = self.tokenizer.encode(" ", add_special_tokens=False)
            token_ids = token_ids[:max_tokens]
            token_count += len(token_ids)
            # The backbone returns hidden states after the final RMSNorm.
            hidden = self.model.model(mx.array([token_ids]), cache=None)
            vector = hidden[0, -1].astype(mx.float32)
            vector /= mx.maximum(mx.linalg.norm(vector), 1e-12)
            embeddings.append(vector.tolist())
            mx.clear_cache()
        return embeddings, token_count
