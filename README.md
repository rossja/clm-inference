# clm-inference
inference engine for [contrastive language models](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B).
we're using the [czl mlx](https://huggingface.co/czl/CLM-v0.1-8B-MLX-8bit) quant here.

## Run

Requires an Apple Silicon Mac, Python 3.11–3.13, and [uv](https://docs.astral.sh/uv/).
The weights are about 9.3 GB; allow additional unified memory for inference.

```sh
git clone git@github.com:rossja/clm-inference.git
cd clm-inference
uv sync
uv run clm-inference
```

The first start downloads the model into Hugging Face's shared cache
(`~/.cache/huggingface/hub` by default). Later starts reuse it. No model files
are stored in this repository. Standard Hub settings such as `HF_HOME`,
`HF_HUB_CACHE`, `HF_TOKEN`, and `HF_HUB_OFFLINE` are respected. An optional
`.env` file in the working directory overrides environment variables.

The server listens on `http://127.0.0.1:8092`. It loads the encoder once and
serializes inference on one worker thread. Inputs in a batch are processed
individually to keep memory use predictable. Change settings in
`config/default.yaml`, or copy it to a local file and run
`uv run clm-inference --config config/local.yaml`.

## API

```sh
curl http://127.0.0.1:8092/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"czl/CLM-v0.1-8B-MLX-8bit","input":["What causes tides on Earth?"]}'
```

`POST /v1/embeddings` accepts a string or up to 32 strings and returns
OpenAI-compatible JSON with one 4096-dimensional embedding per input, in input
order. It uses the backbone's last-token hidden state, normalized to unit
length and returned as float32 values. Texts keep their first 2048 tokens;
`truncate_prompt_tokens` can request a lower limit. Empty strings are encoded
as a single space. Token usage counts reflect the inputs after truncation.
Only `encoding_format: "float"` is supported; token-ID inputs are not supported.

`GET /health` reports readiness after loading. Interactive API documentation
is at `/docs`; the checked-in OpenAPI schema is [swagger.json](swagger.json).

This serves the **embedding encoder**. It does not generate text or apply
CLM's projection heads to rank candidates. A CLM decision service can use
`http://127.0.0.1:8092/v1/embeddings` as its encoder endpoint.

## Development

```sh
uv run pytest
uv run pylint --rcfile="$HOME/src/dotfiles/private/agents/rules/lib/pylintrc" src/clm_inference
```
