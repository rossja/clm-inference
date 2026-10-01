"""Send an example JSON request to the generic CLM inference API."""

import argparse
import json
from pathlib import Path
from urllib import error, request


def main() -> None:
    """Read a caller-owned request and print the server's response."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_file", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8092")
    args = parser.parse_args()
    body = json.loads(args.request_file.read_text(encoding="utf-8"))
    endpoint = "rank" if "answers" in body else "systemone"
    server_url = args.url.rstrip("/")
    http_request = request.Request(
        f"{server_url}/v1/{endpoint}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=120) as response:
            print(json.dumps(json.load(response), indent=2))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        parser.exit(1, f"HTTP {exc.code}: {detail}\n")
    except error.URLError as exc:
        parser.exit(1, f"Cannot reach the inference server: {exc.reason}\n")


if __name__ == "__main__":
    main()
