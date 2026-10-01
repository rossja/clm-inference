"""Exercise the selected backend through the public HTTP API."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import base64
import json
import math
from pathlib import Path
import struct
import time
from urllib.request import Request, urlopen


def main() -> None:
    """Run the experiment and save measurements as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo
    base_url = args.url.rstrip("/")

    def post(endpoint, body):
        """Send JSON to the running inference service."""
        request = Request(
            f"{base_url}/v1/{endpoint}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=180) as response:
            return json.load(response)

    results = {}
    for path in sorted((root / "examples").glob("*.json")):
        body = json.loads(path.read_text())
        endpoint = "rank" if "answers" in body else "systemone"
        start = time.perf_counter()
        response = post(endpoint, body)
        results[path.stem] = {
            "response": response,
            "seconds": time.perf_counter() - start,
        }
        print(path.stem, json.dumps(results[path.stem]), flush=True)

    texts = [
        "hello",
        "",
        "你好，世界",
        "A somewhat longer sentence for mixed lengths.",
    ]
    vectors = post("embeddings", {"input": texts})
    assert len(vectors["data"]) == len(texts)
    for index, item in enumerate(vectors["data"]):
        assert item["index"] == index
        assert len(item["embedding"]) == 4096
        assert all(math.isfinite(x) for x in item["embedding"])
        assert abs(sum(x * x for x in item["embedding"]) ** 0.5 - 1) < 1e-5
    encoded = post("embeddings", {"input": texts, "encoding_format": "base64"})
    for first, second in zip(vectors["data"], encoded["data"]):
        decoded = struct.unpack("<4096f", base64.b64decode(second["embedding"]))
        assert list(decoded) == first["embedding"]
    assert (
        post(
            "embeddings", {"input": "hello world", "truncate_prompt_tokens": 1}
        )["usage"]["total_tokens"]
        == 1
    )
    long = post("embeddings", {"input": "hello " * 2500})
    assert long["usage"]["total_tokens"] == 2048
    batch = post("embeddings", {"input": ["hello"] * 32})
    assert len(batch["data"]) == 32
    assert batch["usage"]["total_tokens"] == 32
    body = json.loads((root / "examples/prompt_injection.json").read_text())
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        concurrent = list(pool.map(lambda _: post("systemone", body), range(8)))
    elapsed = time.perf_counter() - start
    assert all(response == concurrent[0] for response in concurrent)
    report = {
        "examples": results,
        "checks": {
            "finite_normalized_4096_dimensions": True,
            "empty_and_unicode": True,
            "base64_float32_equivalence": True,
            "one_token_truncation": True,
            "2048_token_limit": True,
            "32_text_batch": True,
            "eight_concurrent_consistent_requests": True,
            "eight_concurrent_seconds": elapsed,
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("HTTP checks passed", json.dumps(report["checks"]), flush=True)


if __name__ == "__main__":
    main()
