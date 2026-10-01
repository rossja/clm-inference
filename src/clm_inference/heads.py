"""MLX implementation of CLM's supplied state and action projection heads.

Architecture follows upstream CLM, with PyTorch-compatible exact GELU:
https://github.com/Contrastive-LM/CLM/blob/main/src/clm/heads.py
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class Head(nn.Module):
    """Projection MLP built from the checkpoint's architecture metadata."""

    def __init__(self, config: dict):
        super().__init__()
        width = config["width"]
        self.inp = nn.Linear(config["hidden_size"], width)
        self.hidden = [
            nn.Linear(width, width) for _ in range(config["depth"] - 2)
        ]
        self.norms = [
            nn.LayerNorm(width) if config["layernorm"] else nn.Identity()
            for _ in self.hidden
        ]
        self.out = nn.Linear(width, config["projection_dim"])
        self.act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[
            config["activation"]
        ]()
        self.residual = config["residual"]

    def __call__(self, inputs: mx.array) -> mx.array:
        hidden = self.act(self.inp(inputs))
        for linear, norm in zip(self.hidden, self.norms):
            output = self.act(norm(linear(hidden)))
            hidden = hidden + output if self.residual else output
        return self.out(hidden)


class HeadPair(nn.Module):
    """Load both heads strictly from safetensors, with no PyTorch dependency."""

    def __init__(self, config_path: Path, weights_path: Path):
        super().__init__()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.state_head = Head(config["state_head"])
        self.action_head = Head(config["action_head"])
        self.logit_scale = mx.array(config["logit_scale"], dtype=mx.float32)
        self.load_weights(str(weights_path), strict=True)
        mx.eval(self.parameters())
        self.scale = min(float(mx.exp(self.logit_scale).item()), 100.0)

    def logits(
        self, states: list[list[float]], actions: list[list[float]]
    ) -> list[list[float]]:
        """Return scaled cosine similarities for every state/action pair."""
        states = self.state_head(mx.array(states, dtype=mx.float32))
        actions = self.action_head(mx.array(actions, dtype=mx.float32))
        states /= mx.maximum(mx.linalg.norm(states, axis=-1, keepdims=True),
                             1e-12)
        actions /= mx.maximum(mx.linalg.norm(actions, axis=-1, keepdims=True),
                              1e-12)
        return (self.scale * (states @ actions.T)).tolist()
