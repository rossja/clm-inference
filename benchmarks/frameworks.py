"""Measure complete CLM decisions with an interchangeable encoder backend."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import resource
import statistics
import sys
import time


def main() -> None:
    """Run the experiment and save measurements as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        required=True,
        choices=["mlx-lm", "mlx-lm-batched", "vllm-mlx", "vllm-metal"],
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    import yaml

    settings = yaml.safe_load(args.config.read_text())
    if args.gpu_memory_utilization is None:
        args.gpu_memory_utilization = settings["gpu_memory_utilization"]
    sys.path.insert(0, str(args.repo / "src"))

    import mlx.core as mx
    from clm_inference.engine import Engine
    from clm_inference.heads import HeadPair
    from clm_inference.schema import (
        DecisionRequest,
        RankRequest,
        candidates,
        state_text,
    )

    class AlternativeEncoder:
        """Wrap the alternative frameworks public embedding APIs."""

        def __init__(self):
            if args.backend == "vllm-mlx":
                from vllm_mlx.embedding import EmbeddingEngine

                self.backend = EmbeddingEngine(
                    str(args.model_path),
                    max_length_ceiling=settings["max_tokens"],
                )
                self.backend.load()
            else:
                from vllm import LLM

                self.backend = LLM(
                    model=str(args.model_path),
                    runner="pooling",
                    dtype="bfloat16",
                    max_model_len=settings["max_tokens"],
                    max_num_seqs=settings["max_batch_size"],
                    enforce_eager=True,
                    enable_prefix_caching=False,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                )

        def embed(self, texts, max_tokens):
            """Encode bounded inputs and report their token use."""
            if args.backend == "vllm-mlx":
                vectors = self.backend.embed(texts)
                tokens = self.backend.count_tokens(texts)
            else:
                tokenizer = self.backend.get_tokenizer()
                prompts = []
                for text in texts:
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    if not ids:
                        ids = tokenizer.encode(" ", add_special_tokens=False)
                    prompts.append({"prompt_token_ids": ids[:max_tokens]})
                outputs = self.backend.embed(prompts, use_tqdm=False)
                vectors = [item.outputs.embedding for item in outputs]
                tokens = sum(len(item.prompt_token_ids) for item in outputs)
            return vectors, tokens

    started = time.perf_counter()
    if args.backend in ("mlx-lm", "mlx-lm-batched"):
        from mlx_lm import load
        import mlx.nn as nn

        class BaselineEncoder:
            """Preserve the serial MLX encoder used before the framework switch."""

            def __init__(self, model_path):
                self.model, self.tokenizer = load(str(model_path), lazy=True)
                self.model.lm_head = nn.Identity()
                mx.eval(self.model.parameters())

            def embed(self, inputs, max_tokens):
                """Pool the final token and normalize in float32."""
                vectors, tokens = [], 0
                for text in inputs:
                    ids = self.tokenizer.encode(text, add_special_tokens=False)
                    if not ids:
                        ids = self.tokenizer.encode(
                            " ", add_special_tokens=False
                        )
                    ids = ids[:max_tokens]
                    tokens += len(ids)
                    hidden = self.model.model(mx.array([ids]), cache=None)
                    vector = hidden[0, -1].astype(mx.float32)
                    vector /= mx.maximum(mx.linalg.norm(vector), 1e-12)
                    vectors.append(vector.tolist())
                    mx.clear_cache()
                return vectors, tokens

        encoder = BaselineEncoder(args.model_path)
        if args.backend == "mlx-lm-batched":

            def batched_embed(texts, max_tokens):
                """Run causal right-padded batches with a padded-token budget."""
                ids = [
                    encoder.tokenizer.encode(
                        text or " ", add_special_tokens=False
                    )[:max_tokens]
                    for text in texts
                ]
                groups = []
                current = []
                longest = 0
                for row in ids:
                    prospective = max(longest, len(row))
                    if (
                        current
                        and prospective * (len(current) + 1)
                        > settings["max_tokens"] * 2
                    ):
                        groups.append(current)
                        current, longest = [], 0
                    current.append(row)
                    longest = max(longest, len(row))
                if current:
                    groups.append(current)
                vectors = []
                for group in groups:
                    width = max(map(len, group))
                    inputs = mx.array(
                        [row + [0] * (width - len(row)) for row in group]
                    )
                    hidden = encoder.model.model(inputs, cache=None)
                    pooled = hidden[
                        mx.arange(len(group)),
                        mx.array([len(row) - 1 for row in group]),
                    ].astype(mx.float32)
                    pooled /= mx.maximum(
                        mx.linalg.norm(pooled, axis=-1, keepdims=True), 1e-12
                    )
                    vectors.extend(pooled.tolist())
                    mx.clear_cache()
                return vectors, sum(map(len, ids))

            encoder.embed = batched_embed
    else:
        encoder = AlternativeEncoder()
    heads = HeadPair(args.checkpoint)
    engine = Engine(encoder, heads, settings["model"], settings["max_tokens"])
    loaded = time.perf_counter() - started
    examples = []
    texts = []
    for path in sorted((args.repo / "examples").glob("*.json")):
        body = json.loads(path.read_text())
        ranking = "answers" in body
        req = RankRequest(**body) if ranking else DecisionRequest(**body)
        examples.append((path.stem, ranking, req))
        if ranking:
            texts.append(state_text(req.context, req.question))
            texts.extend(req.answers)
        else:
            for q in req.questions.values():
                texts.append(state_text(req.state, q.instructions))
                texts.extend(candidates(q)[1])

    results = {}
    for name, ranking, req in examples:
        t0 = time.perf_counter()
        output = engine.rank(req) if ranking else engine.answer(req)
        results[name] = {
            "response": output.model_dump(),
            "first_request_seconds": time.perf_counter() - t0,
        }
        print(name, json.dumps(results[name]), flush=True)

    timings = {name: [] for name, _, _ in examples}
    for iteration in range(args.repeats):
        for name, ranking, req in examples:
            t0 = time.perf_counter()
            engine.rank(req) if ranking else engine.answer(req)
            timings[name].append(time.perf_counter() - t0)
        print("repeat", iteration + 1, flush=True)

    batch_times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        vectors, tokens = encoder.embed(texts, settings["max_tokens"])
        batch_times.append(time.perf_counter() - t0)

    # Mixed-length and singleton consistency detects pooling/padding regressions.
    probe_texts = [
        texts[0],
        texts[-1],
        "hello",
        "A longer sentence to exercise padding.",
    ]
    batch, _ = encoder.embed(probe_texts, settings["max_tokens"])
    single = [
        encoder.embed([text], settings["max_tokens"])[0][0]
        for text in probe_texts
    ]
    consistency = max(
        abs(a - b) for v, w in zip(batch, single) for a, b in zip(v, w)
    )
    versions = {}
    for name in [
        "mlx",
        "mlx-lm",
        "mlx-embeddings",
        "vllm-mlx",
        "vllm-metal",
        "vllm",
        "torch",
        "transformers",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = {
        "backend": args.backend,
        "versions": versions,
        "load_seconds": loaded,
        "gpu_memory_utilization": args.gpu_memory_utilization
        if args.backend == "vllm-metal"
        else None,
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mlx_peak_bytes": mx.get_peak_memory(),
        "results": results,
        "request_seconds": timings,
        "median_request_seconds": statistics.median(
            [x for values in timings.values() for x in values]
        ),
        "batch_encoder_seconds": batch_times,
        "batch_text_count": len(texts),
        "batch_tokens": tokens,
        "mixed_length_singleton_max_abs_difference": consistency,
        "probe_texts": probe_texts,
        "probe_vectors": batch,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "saved",
        args.output,
        "median request seconds",
        report["median_request_seconds"],
        flush=True,
    )


if __name__ == "__main__":
    main()
