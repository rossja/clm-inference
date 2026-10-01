"""Compare MLX projection heads with an independent NumPy implementation."""

import json
import math

import mlx.core as mx
import numpy as np
import pytest

from clm_inference.heads import HeadPair


def reference_projection(inputs, weights, prefix, residual):
    """Evaluate the upstream linear/GELU/LayerNorm architecture independently."""
    def linear(values, name):
        return values @ weights[f"{prefix}.{name}.weight"].T + weights[
            f"{prefix}.{name}.bias"
        ]

    def gelu(values):
        return values * (1 + np.vectorize(math.erf)(values / math.sqrt(2))) / 2

    hidden = gelu(linear(inputs, "inp"))
    output = linear(hidden, "hidden.0")
    output = (output - output.mean(axis=-1, keepdims=True)) / np.sqrt(
        output.var(axis=-1, keepdims=True) + 1e-5
    )
    output = output * weights[f"{prefix}.norms.0.weight"] + weights[
        f"{prefix}.norms.0.bias"
    ]
    output = gelu(output)
    hidden = hidden + output if residual else output
    result = linear(hidden, "out")
    return result / np.maximum(np.linalg.norm(result, axis=-1, keepdims=True),
                               1e-12)


@pytest.mark.parametrize("residual", [False, True])
def test_heads_match_reference_and_scale_clamp(tmp_path, residual):
    rng = np.random.default_rng(42)
    config = {
        "hidden_size": 4, "width": 3, "projection_dim": 2, "depth": 3,
        "activation": "gelu", "layernorm": True, "residual": residual,
    }
    shapes = {
        "inp.weight": (3, 4), "inp.bias": (3,),
        "hidden.0.weight": (3, 3), "hidden.0.bias": (3,),
        "norms.0.weight": (3,), "norms.0.bias": (3,),
        "out.weight": (2, 3), "out.bias": (2,),
    }
    weights = {
        f"{prefix}.{name}": rng.standard_normal(shape).astype(np.float32)
        for prefix in ("state_head", "action_head")
        for name, shape in shapes.items()
    }
    weights["logit_scale"] = np.array(math.log(101), dtype=np.float32)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "state_head": config, "action_head": config, "logit_scale": 0,
    }), encoding="utf-8")
    weights_path = tmp_path / "heads.safetensors"
    mx.save_safetensors(str(weights_path), {
        key: mx.array(value) for key, value in weights.items()
    })
    pair = HeadPair(config_path, weights_path)
    assert pair.scale == 100
    states = rng.standard_normal((2, 4)).astype(np.float32)
    actions = rng.standard_normal((3, 4)).astype(np.float32)
    expected = 100 * (
        reference_projection(states, weights, "state_head", residual)
        @ reference_projection(actions, weights, "action_head", residual).T
    )
    actual = pair.logits(states.tolist(), actions.tolist())
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)
    weights.pop("state_head.inp.bias")
    mx.save_safetensors(str(weights_path), {
        key: mx.array(value) for key, value in weights.items()
    })
    with pytest.raises(ValueError):
        HeadPair(config_path, weights_path)
