"""MLX implementation of CLM's supplied state and action projection heads.

Architecture follows upstream CLM, with PyTorch-compatible exact GELU:
https://github.com/Contrastive-LM/CLM/blob/main/src/clm/heads.py
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import torch


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
    """Read the official PyTorch checkpoint and evaluate both heads in MLX."""

    def __init__(self, weights_path: Path):
        super().__init__()
        checkpoint = torch.load(
            weights_path, map_location="cpu", weights_only=True
        )
        config = checkpoint["cfg"]
        self.encoder_model = config["model"]
        self.state_head = Head(config)
        self.action_head = Head(config)
        self.logit_scale = mx.array(0, dtype=mx.float32)
        weights = [
            (f"{prefix}.{name}", mx.array(value.numpy()))
            for prefix in ("state_head", "action_head")
            for name, value in checkpoint[prefix].items()
        ]
        weights.append(("logit_scale", mx.array(
            checkpoint["logit_scale"].numpy()
        )))
        self.load_weights(weights, strict=True)
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
