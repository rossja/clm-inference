# Apple Silicon framework selection

The selection criterion is correct CLM inference followed by low latency and
good batch throughput on the available M3 Max with 48 GiB unified memory.
Memory use, dependency size, startup time and implementation complexity also
matter. This evaluates the official Qwen3-8B BF16 checkpoint, not token
generation performance or a different Qwen embedding model.

We selected **vLLM Metal 0.30.0**. It ran all six client examples, reduced
median inference time from 328 ms to 192 ms, and processed the combined
24-text encoder workload in 0.81 seconds. vllm-mlx measured 226 ms and
2.60 seconds. A batched direct mlx-lm experiment measured 235 ms and 2.84
seconds. The original serial mlx-lm encoder needed 1.87 seconds for that batch.

The Metal configuration uses `gpu_memory_utilization: 0.5`. Its peak MLX
allocation was 18.1 GiB and peak process RSS was 19.2 GiB. vllm-mlx used
14.8 GiB in MLX and 14.9 GiB RSS. Metal initially used a 0.6 budget and
21.9 GiB in MLX; reducing the cache allocation preserved its speed. The chosen
configuration loaded in about 20 seconds after imports, compared with about
12 seconds for vllm-mlx and 3 seconds for the original direct MLX run.
Startup measurements include framework initialization and warmup; they are
affected by existing filesystem and shader caches and are not cold-start trials.

## Method

Each framework ran alone in an isolated Python 3.12 environment. We used the
same cached encoder revision, original CLM checkpoint, local MLX projection
heads, question formatting, candidate descriptions and temperature. The
checkpoint revisions, versions, example hashes, individual timing samples
and responses are in [framework-comparison.json](framework-comparison.json).

The first request for each example warmed the runtime and was excluded from
the latency median. Baselines then ran three repetitions of each of the six
examples; the selected Metal configuration ran seven. The reported request
median covers the resulting 18 or 42 in-process measurements, including
tokenization, encoding, projections and answer formatting. It excludes HTTP
transport, loading and first requests. Batch timing covers all 24 encoder
texts together: 631 actual tokens with mixed lengths. Prefix caching was off
for Metal, and the direct and vllm-mlx paths did not cache embeddings.

The direct batched experiment right-pads token inputs, selects each text's
last real token, normalizes in float32, and limits each batch to twice the
configured token limit in padded token positions. It tests whether batching
alone closes the gap without replacing Apple's mlx-lm framework.

All frameworks returned the same choice winners and ranking on the six
examples. Their numerical outputs differ. The four probe embeddings had
cosine similarity above 0.99995 against the serial baseline, but this small
probe is not a general equivalence test. Both batched frameworks and the
batched direct experiment showed some differences between batched and
singleton embeddings. Those differences can affect classification
probabilities near a decision boundary. The three known example errors
persisted with every framework.

The implementation retains the generic API and CLM heads. vLLM Metal now owns
encoder loading, scheduling, attention, pooling and normalization. The adapter
only tokenizes inputs, applies our documented prefix truncation and empty-input
rule, and collects vectors. The default is single-process Metal mode. The
HTTP worker serializes requests; continuous batching across independent HTTP
requests was not implemented or benchmarked. Eight concurrent requests were
verified to complete consistently through that queue.

## Shutdown compatibility

The first explicit vLLM shutdown segfaulted in
`torch.accelerator.empty_host_cache()` with the pinned PyTorch 2.13.0 release.
We isolated it from inference: allocating a one-element MPS tensor and calling
that function also exited with code 139. vLLM's distributed cleanup invokes
the failing function after its Metal worker has already shut down.

The adapter temporarily routes that one cleanup call to
`torch.mps.empty_cache()` while executing the normal engine shutdown. The
original PyTorch function is restored afterward, including on exceptions.
This workaround affects teardown only; model computation and the framework's
other cleanup steps remain unchanged. Unit tests cover routing and restoration,
and real HTTP startup, inference and shutdown are checked separately. Revisit
the workaround when upgrading the pinned framework and PyTorch versions.

The minimal failing reproduction is:

```sh
uv run python -c 'import torch; tensor = torch.ones(1, device="mps"); torch.accelerator.empty_host_cache()'
```

[PyTorch's pinned implementation](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/accelerator/memory.py)
calls the native host-cache function directly. The local reproduction is the
evidence for this crash; the separately reported
[MPS accelerator cache issue](https://github.com/pytorch/pytorch/issues/186430)
concerns the related `empty_cache()` API and is not an exact reproduction of
our failure.

## Reproduce

The benchmark is [../benchmarks/frameworks.py](../benchmarks/frameworks.py).
It accepts paths to the existing shared-cache snapshots, a configuration file,
an output file, a backend, and a repetition count. It never creates a local
copy of the weights. Run it with one model process at a time; stop the HTTP
server before measuring another framework.

From the project root, obtain the recorded snapshots in the shared cache:

```sh
CLM_ENCODER_PATH=$(uv run hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --quiet)
CLM_HEAD_PATH=$(uv run hf download Contrastive-LM/CLM-v0.1-8B CLM_v0.1-8B.pt --revision e939398d4556fcd9400c76fa8c5a513202f42b0a --quiet)
```

Run the selected framework using the project's locked environment:

```sh
VLLM_ENABLE_V1_MULTIPROCESSING=0 uv run python benchmarks/frameworks.py \
  --backend vllm-metal --repo "$PWD" --config config/default.yaml \
  --model-path "$CLM_ENCODER_PATH" --checkpoint "$CLM_HEAD_PATH" \
  --output /tmp/clm-metal-benchmark.json --repeats 7
```

For vllm-mlx, create an isolated environment and install the tested source
revision. Its major dependency versions are recorded in the saved result;
transitive dependencies can change unless they are pinned separately.

```sh
uv venv --python 3.12 /tmp/clm-benchmark-mlx
uv pip install --python /tmp/clm-benchmark-mlx/bin/python \
  'vllm-mlx @ git+https://github.com/waybarrios/vllm-mlx@f5d7e00a5c23d7dd478bc07af03cc53fea75e6be'
/tmp/clm-benchmark-mlx/bin/python benchmarks/frameworks.py \
  --backend vllm-mlx --repo "$PWD" --config config/default.yaml \
  --model-path "$CLM_ENCODER_PATH" --checkpoint "$CLM_HEAD_PATH" \
  --output /tmp/clm-mlx-benchmark.json --repeats 3
```

For the original direct and batched baselines, use a separate environment with
the recorded MLX versions. The script contains the original serial adapter so
future production backend changes do not alter the baseline.

```sh
uv venv --python 3.12 /tmp/clm-benchmark-direct
uv pip install --python /tmp/clm-benchmark-direct/bin/python \
  'mlx==0.32.3' 'mlx-lm==0.31.3' 'torch==2.14.1' 'pydantic==2.13.5' 'pyyaml==6.0.3'
/tmp/clm-benchmark-direct/bin/python benchmarks/frameworks.py \
  --backend mlx-lm --repo "$PWD" --config config/default.yaml \
  --model-path "$CLM_ENCODER_PATH" --checkpoint "$CLM_HEAD_PATH" \
  --output /tmp/clm-direct-benchmark.json --repeats 3
```

Repeat that command with `--backend mlx-lm-batched` for the batching experiment.
The initial Metal trial can be repeated with
`--gpu-memory-utilization 0.6`. Otherwise the script reads its memory budget
and serving limits from the selected configuration file.

With the production HTTP server running, repeat its integration checks:

```sh
uv run python benchmarks/http_check.py --repo "$PWD" \
  --url http://127.0.0.1:8092 --output /tmp/clm-http-check.json
```

The saved [HTTP checks](framework-http-check.json) cover the six examples,
finite unit-length vectors, 4096 dimensions, input order, empty and Unicode
inputs, float/base64 equivalence, one-token and 2048-token truncation, a
32-text batch and eight concurrent requests. These checks establish that the
API remains usable, not that the model is an accurate security detector.

## Framework sources

- [vLLM Metal installation](https://github.com/vllm-project/vllm-metal/blob/v0.30.0/docs/installation.md)
- [vLLM Metal text pooling](https://github.com/vllm-project/vllm-metal/blob/v0.30.0/docs/text_embedding_pooling.md)
- [vllm-mlx embedding implementation at the tested revision](https://github.com/waybarrios/vllm-mlx/blob/f5d7e00a5c23d7dd478bc07af03cc53fea75e6be/vllm_mlx/embedding.py)
