# clm-inference
inference engine for [contrastive language models](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B).
Uses the official CLM heads with the full-precision
[Qwen3-8B encoder](https://huggingface.co/Qwen/Qwen3-8B), served through
[vLLM Metal](https://github.com/vllm-project/vllm-metal) on Apple Silicon.

## Run

Requires an Apple Silicon Mac with macOS 15 or later, Python 3.12, and
[uv](https://docs.astral.sh/uv/). The dependency lock uses the official matched
vLLM and vLLM Metal 0.30.0 wheels for macOS arm64.
The encoder weights are about 16.4 GB and the heads about 76 MB. Allow
additional unified memory for inference; the full-precision backend was tested
on a Mac with 48 GB of unified memory.

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
trained projection heads once. vLLM Metal runs the encoder on the GPU using
MLX and native Metal attention kernels; the CLM heads run in MLX. Loading,
inference and engine shutdown use one owning worker thread.
The encoder comes directly from `Qwen/Qwen3-8B` in bfloat16, without quantization.
The heads come from `Contrastive-LM/CLM-v0.1-8B/CLM_v0.1-8B.pt`. PyTorch reads
that checkpoint on the CPU with `weights_only=True`; its tensors are copied
into MLX in memory, without creating converted model files. Both encoder and
heads use the shared Hugging Face cache. The loader checks that the heads match
the configured encoder. Texts are submitted together to vLLM's scheduler for
last-token pooling and L2 normalization. `gpu_memory_utilization` controls the
framework's memory budget and defaults to 0.5 for the tested 48 GiB Mac.
Smaller-memory machines may need a different budget; BF16 weights still need
about 16 GB before cache and working memory. The adapter defaults to vLLM's
documented single-process Metal mode; an explicit
`VLLM_ENABLE_V1_MULTIPROCESSING` environment value is respected. Change settings in
`config/default.yaml`, or copy it to a local file and run
`uv run clm-inference --config config/local.yaml`.

## Framework selection

The goal is the best Apple Silicon framework for this CLM workload. We measured
the same official BF16 weights and six unchanged requests on an M3 Max with
48 GiB unified memory, keeping the CLM heads identical across frameworks.

| Encoder framework | Median inference per request | 24-text encoder batch | Peak MLX memory |
| --- | --- | --- | --- |
| Original serial `mlx-lm` | 328 ms | 1.87 s | 14.3 GiB |
| Batched `mlx-lm` experiment | 235 ms | 2.84 s | 14.7 GiB |
| `vllm-mlx` 0.5.0 | 226 ms | 2.60 s | 14.8 GiB |
| vLLM Metal 0.30.0, budget 0.5 | **192 ms** | **0.81 s** | 18.1 GiB |

vLLM Metal is the selected default: about 42% lower median latency than the
original backend and about 17% lower than vllm-mlx, with substantially better
batch throughput. It uses more memory, starts more slowly, and has a larger
dependency set. The batched MLX experiment avoids attributing the entire gain
to comparing a batched framework with a serial implementation.

The pinned PyTorch release segfaults in its generic MPS host-cache cleanup.
Our adapter applies a scoped workaround during vLLM shutdown, using the MPS
cache API and restoring the original function afterward. This does not alter
inference. Details and the reproduction are in the framework report.

These are in-process inference measurements, excluding model loading and first
requests, with three repetitions per baseline and seven for the selected
configuration. The batch contains 24 mixed-length texts and 631 real tokens.
All backends produced the same winning answers, but probabilities and some
embeddings varied with batching and BF16 kernel arithmetic. This is a measured
choice for this hardware and workload, not a claim about every Apple Silicon
model or workload. The model's previously observed classification failures
persisted. The HTTP service still serializes requests on its worker; this
benchmark does not claim continuous batching across separate HTTP requests.

[Benchmark details and reproduction](docs/framework-selection.md),
[raw measurements](docs/framework-comparison.json), and
[HTTP verification](docs/framework-http-check.json) are saved in the repository.

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

Both endpoints accept an optional `model` equal to the configured CLM model ID
(`Contrastive-LM/CLM-v0.1-8B` by default)
and a `temperature` in `(0, 100]`, defaulting to 1. Higher temperature makes
the distribution flatter. There is no task-specific model selection.
`GET /v1/models` lists the CLM model and encoder separately.

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

The same six example requests were run with the previous
[8-bit backend](https://huggingface.co/czl/CLM-v0.1-8B-MLX-8bit) and the official
full-precision backend. No instructions or candidate descriptions were changed.

| Historical example result | Previous MLX 8-bit | Direct MLX (BF16) | Upstream CPU vLLM + PyTorch (BF16) |
| --- | --- | --- | --- |
| Support routing | Security: 94.00% | Security: 93.53% | Security: 93.30% |
| Phishing email | Spam: 78.96% (incorrect) | Spam: 78.08% (incorrect) | Spam: 77.85% (incorrect) |
| Prompt injection | Benign: 98.57% (incorrect) | Benign: 98.98% (incorrect) | Benign: 98.67% (incorrect) |
| Resolved case needs follow-up | True: 98.80% (incorrect) | True: 98.71% (incorrect) | True: 98.73% (incorrect) |
| Urgency score | 2.3679 out of 3 | 2.4263 out of 3 | 2.3968 out of 3 |
| Answer ranking | Correct explanation first: 99.998% | Correct explanation first: 99.998% | Correct explanation first: 99.999% |

Removing quantization did not correct the three classification errors in these
examples. On the full-precision run, the MLX heads matched the official CLM
PyTorch head implementation within 0.000002 in probability, using the same
encoder outputs. This checks projection math and request formatting, rather
than equivalence of the MLX and vLLM encoders. This table records the earlier
backend comparison; the current Metal results are in the framework measurements.

A separate complete upstream run on 2026-09-30 used the unmodified CLM HTTP
server and PyTorch heads with native vLLM 0.11.0 on the Mac's CPU. Qwen3-8B ran
in BF16 with last-token pooling and a 2048-token limit. All six unchanged
requests completed, and the same three classification errors persisted.
Probabilities differ between the encoder implementations; this demonstrates
the same example failures, not exact numerical parity or a CUDA runtime check.
[Saved responses and runtime versions](docs/upstream-comparison.json) include
the MLX baseline, upstream outputs and hashes of the input files. To replay the
examples against an upstream CLM server, use the existing runner:

```bash
uv run python examples/run.py examples/prompt_injection.json --url http://127.0.0.1:8700
```

The security examples need further evaluation before practical use.

## Embeddings and API documentation

`POST /v1/embeddings` accepts a string or up to 32 strings and returns
OpenAI-compatible JSON with one 4096-dimensional embedding per input, in input
order. It uses the backbone's last-token hidden state, normalized to unit
length and returned as float32 values. Texts keep their first 2048 tokens;
`truncate_prompt_tokens` can request a lower limit. Empty strings are encoded
as a single space. Token usage counts reflect the inputs after truncation.
`encoding_format` supports `float` (default) and `base64` (little-endian float32,
as requested by upstream CLM clients). Token-ID inputs are not supported.
For this endpoint, an optional `model` identifies the encoder
(`Qwen/Qwen3-8B` by default), rather than the CLM heads.

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
ordering, encoder token boundaries and lifecycle, and projection heads against
independent NumPy calculations. Head tests require Metal access. The real HTTP
verification also checks last-token embedding dimensions, normalization,
truncation, base64 encoding, empty inputs and concurrent requests.
