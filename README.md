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

The server listens on `http://127.0.0.1:8092`. It loads the encoder and both
trained projection heads once, and runs them in MLX on one worker thread.
The heads use the safetensors supplied with the MLX checkpoint; no PyTorch
installation or checkpoint conversion is needed. Both encoder and heads use
the shared Hugging Face cache. Inputs in a batch are processed
individually to keep memory use predictable. Change settings in
`config/default.yaml`, or copy it to a local file and run
`uv run clm-inference --config config/local.yaml`.

## Decisions and classification

The engine is generic and zero-shot. Each request supplies the context,
instructions and possible answers. Application labels, routing rules and
security definitions live in the calling application. The engine applies
CLM's trained state and action heads and computes a distribution over those
answers; it does not train a classifier for each task.

`POST /v1/systemone` accepts a `state` and a map of typed `questions`:

```sh
curl http://127.0.0.1:8092/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"A customer was charged twice.","questions":{"route":{"type":"choice","instructions":"Which team should handle this case?","criteria":{"billing":"Charges, invoices and refunds","technical":"Product defects and outages"}}}}'
```

The question and answer formats follow [upstream CLM](https://github.com/Contrastive-LM/CLM#api-reference):

| Type | Request criteria | Returned answer |
| --- | --- | --- |
| `choice` | Object mapping labels to descriptions | Winning `choice`, `confidence`, and `probabilities` for every label |
| `noul` | Optional object with `true` and `false` descriptions | `noul`: probability the statement in `instructions` is true; false has probability `1 - noul` |
| `score` | Ordered list of at least two level descriptions | `score`: expected level index, `confidence`, `legend`, and per-level `probabilities` |

Every question requires `instructions` and `type`. Responses preserve question
identifiers under `answers` and include token usage. `state` can be text, an
object or an array; structured states are rendered as prose, matching upstream
CLM. For choice questions, the action head sees each description directly,
falling back to its label when the description is empty or null. The state
head sees the context followed by the instructions.

`POST /v1/rank` accepts `context`, `question`, and a non-empty `answers` list.
It returns `ranked` candidates, best first, with a one-based `rank` and `prob`.
It uses the same trained heads and scoring as `choice`.

Both endpoints accept an optional `model` equal to the configured Hub model ID
and a `temperature` in `(0, 100]`, defaulting to 1. Higher temperature makes
the distribution flatter. There is no task-specific model selection.
`GET /v1/models` identifies the loaded model.

Probabilities are **relative to the supplied candidate set**, not calibrated
estimates of real-world accuracy. `confidence` follows upstream: the highest
probability minus the mean of the other probabilities. A `score` uses zero-based
rubric indices: four levels produce a value from 0 to 3, including fractions.

By default, a decision request may encode up to 32 texts: one context plus
instructions per question, plus all candidate texts. Rank requests count one
context plus all answers. Each text keeps its first 2048 tokens, matching the
model's serving contract. The limits come from `config/default.yaml`.

## Examples

Start the server, then send any of the example request files:

```sh
uv run python examples/run.py examples/support_routing.json
uv run python examples/run.py examples/email_classification.json
uv run python examples/run.py examples/prompt_injection.json
uv run python examples/run.py examples/statement_probability.json
uv run python examples/run.py examples/urgency_score.json
uv run python examples/run.py examples/answer_ranking.json
```

| Example | Output type | Caller-defined task |
| --- | --- | --- |
| [Support routing](examples/support_routing.json) | `choice` | Billing, technical, routine account access, or security incident escalation |
| [Email classification](examples/email_classification.json) | `choice` | Spam, phishing, or benign |
| [Prompt injection](examples/prompt_injection.json) | `choice` | Malicious or benign, with both probabilities |
| [Statement probability](examples/statement_probability.json) | `noul` | Probability a support case still requires troubleshooting |
| [Urgency score](examples/urgency_score.json) | `score` | Expected urgency on an ordered four-level rubric |
| [Answer ranking](examples/answer_ranking.json) | Ranking | Rank proposed answers to a question |

The files are client examples, not server configuration. Edit their text,
instructions and criteria or send your own JSON without changing the engine.
The runner prints the server's response and accepts `--url` for another server.
These examples demonstrate API use; they do not measure classification accuracy
or detector reliability.

In the initial smoke check, support routing selected security and answer ranking
selected the correct explanation. The phishing email was incorrectly labeled
spam, the injection was incorrectly labeled benign, and the resolved support
case was incorrectly judged to need follow-up. The MLX heads matched the
official CLM PyTorch heads on all six requests within 0.000004 in probability,
using the same 8-bit encoder outputs. This verifies the projection implementation;
it does not establish full-precision encoder accuracy or rule out quantization
effects. The security examples need further evaluation before practical use.

## Embeddings and API documentation

`POST /v1/embeddings` accepts a string or up to 32 strings and returns
OpenAI-compatible JSON with one 4096-dimensional embedding per input, in input
order. It uses the backbone's last-token hidden state, normalized to unit
length and returned as float32 values. Texts keep their first 2048 tokens;
`truncate_prompt_tokens` can request a lower limit. Empty strings are encoded
as a single space. Token usage counts reflect the inputs after truncation.
`encoding_format` supports `float` (default) and `base64` (little-endian float32,
as requested by upstream CLM clients). Token-ID inputs are not supported.

`GET /health` reports readiness after loading. Interactive API documentation
is at `/docs`; the checked-in OpenAPI schema is [swagger.json](swagger.json).

The service scores supplied answers and produces typed decisions. It does not
generate new text or candidate actions.

## Development

```sh
uv run pytest
uv run pylint --jobs=1 --rcfile=.pylintrc src/clm_inference examples/run.py
```

The tests check API validation, input formatting, typed answers, candidate
ordering, last-token pooling, and projection heads against independent NumPy
calculations. Pooling and head tests require Metal access.
