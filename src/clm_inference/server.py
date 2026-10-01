"""Serve CLM embeddings through a small OpenAI-compatible HTTP API."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import yaml


class Settings(BaseModel):
    """Server settings supplied by a YAML configuration file."""

    model: str = Field(min_length=1)
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    max_tokens: int = Field(ge=1, le=2048)
    max_batch_size: int = Field(ge=1)


class EmbeddingRequest(BaseModel):
    """Text inputs and the OpenAI options supported by this encoder."""

    input: str | list[str]
    model: str | None = None
    encoding_format: Literal["float"] = "float"
    truncate_prompt_tokens: int | None = Field(default=None, ge=1)


class Embedding(BaseModel):
    """One embedding, indexed in the original input order."""

    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class Usage(BaseModel):
    """Token counts after truncation."""

    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response."""

    object: Literal["list"] = "list"
    data: list[Embedding]
    model: str
    usage: Usage


def load_encoder(model: str):
    """Download into the standard Hub cache and load on the inference thread."""
    # Read .env before Hub caches its environment settings; import MLX here
    # so OpenAPI generation does not need Metal.
    from huggingface_hub import snapshot_download  # pylint: disable=import-outside-toplevel
    from clm_inference.encoder import Encoder  # pylint: disable=import-outside-toplevel

    path = snapshot_download(
        model,
        allow_patterns=[
            "*.json", "model*.safetensors", "tokenizer.model", "*.tiktoken"
        ],
    )
    return Encoder(Path(path))


def create_app(settings: Settings) -> FastAPI:
    """Create an app that loads and evaluates MLX on a single worker thread."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            app.state.executor = executor
            app.state.encoder = await loop.run_in_executor(
                executor, load_encoder, settings.model
            )
            yield

    app = FastAPI(
        title="CLM Inference", version="0.1.0", lifespan=lifespan
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Report readiness after the model has loaded."""
        return {"status": "ok", "model": settings.model}

    @app.post("/v1/embeddings")
    async def embed(request: EmbeddingRequest) -> EmbeddingResponse:
        """Return one normalized 4096-dimensional vector per input text."""
        if request.model is not None and request.model != settings.model:
            raise HTTPException(400, "The requested model is not loaded.")
        texts = (
            [request.input] if isinstance(request.input, str) else request.input
        )
        if not 1 <= len(texts) <= settings.max_batch_size:
            raise HTTPException(
                400, f"Provide 1 to {settings.max_batch_size} texts."
            )
        max_tokens = request.truncate_prompt_tokens or settings.max_tokens
        if max_tokens > settings.max_tokens:
            raise HTTPException(
                400, "truncate_prompt_tokens exceeds max_tokens."
            )
        vectors, tokens = await asyncio.get_running_loop().run_in_executor(
            app.state.executor, app.state.encoder.embed, texts, max_tokens
        )
        return EmbeddingResponse(
            data=[
                Embedding(index=index, embedding=vector)
                for index, vector in enumerate(vectors)
            ],
            model=settings.model,
            usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
        )

    return app


def read_settings(path: Path) -> Settings:
    """Read configuration and apply the working directory's .env file."""
    load_dotenv(Path.cwd() / ".env", override=True)
    with path.open(encoding="utf-8") as config:
        return Settings.model_validate(yaml.safe_load(config))


def main() -> None:
    """Start the server from the repository root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/default.yaml")
    )
    args = parser.parse_args()
    settings = read_settings(args.config)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
