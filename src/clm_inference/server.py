"""Serve CLM zero-shot decisions, ranking and embeddings on Apple Silicon."""

import argparse
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
import struct
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import yaml

from clm_inference.schema import (
    DecisionRequest, DecisionResponse, RankRequest, RankResponse, candidates,
)


class Settings(BaseModel):
    """Server settings supplied by a YAML configuration file."""

    model: str = Field(min_length=1)
    heads_config: str = Field(min_length=1)
    heads_weights: str = Field(min_length=1)
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    max_tokens: int = Field(ge=1, le=2048)
    max_batch_size: int = Field(ge=1)


class EmbeddingRequest(BaseModel):
    """Text inputs and the OpenAI options supported by this encoder."""

    input: str | list[str]
    model: str | None = None
    encoding_format: Literal["float", "base64"] = "float"
    truncate_prompt_tokens: int | None = Field(default=None, ge=1)


class Embedding(BaseModel):
    """One embedding, indexed in the original input order."""

    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float] | str


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


def load_engine(settings: Settings):
    """Load the encoder and heads from the standard Hub cache on one thread."""
    # Read .env before Hub caches its environment settings; import MLX here
    # so OpenAPI generation does not need Metal.
    from huggingface_hub import snapshot_download  # pylint: disable=import-outside-toplevel
    from clm_inference.encoder import Encoder  # pylint: disable=import-outside-toplevel
    from clm_inference.engine import Engine  # pylint: disable=import-outside-toplevel
    from clm_inference.heads import HeadPair  # pylint: disable=import-outside-toplevel

    path = snapshot_download(
        settings.model,
        allow_patterns=[
            "*.json", "model*.safetensors", "tokenizer.model", "*.tiktoken",
            settings.heads_config, settings.heads_weights,
        ],
    )
    path = Path(path)
    return Engine(
        Encoder(path),
        HeadPair(path / settings.heads_config, path / settings.heads_weights),
        settings.model, settings.max_tokens,
    )


def create_app(settings: Settings) -> FastAPI:
    """Create an app that loads and evaluates MLX on a single worker thread."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            app.state.executor = executor
            app.state.engine = await loop.run_in_executor(
                executor, load_engine, settings
            )
            yield

    app = FastAPI(
        title="CLM Inference", version="0.1.0", lifespan=lifespan
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Report readiness after the model has loaded."""
        return {"status": "ok", "model": settings.model}

    def validate_model(model: str | None) -> None:
        if model is not None and model != settings.model:
            raise HTTPException(400, "The requested model is not loaded.")

    @app.get("/v1/models")
    async def models() -> dict[str, list[dict[str, str]]]:
        """Identify the loaded encoder and trained-head model."""
        return {"models": [{"name": settings.model}]}

    @app.post("/v1/systemone")
    async def systemone(request: DecisionRequest) -> DecisionResponse:
        """Answer caller-defined choice, noul and score questions."""
        validate_model(request.model)
        text_count = sum(
            1 + len(candidates(question)[0])
            for question in request.questions.values()
        )
        if text_count > settings.max_batch_size:
            raise HTTPException(
                400, "Questions and options exceed batch limit."
            )
        return await asyncio.get_running_loop().run_in_executor(
            app.state.executor, app.state.engine.answer, request
        )

    @app.post("/v1/rank")
    async def rank(request: RankRequest) -> RankResponse:
        """Rank arbitrary candidate answers using the trained CLM heads."""
        validate_model(request.model)
        if 1 + len(request.answers) > settings.max_batch_size:
            raise HTTPException(400, "Context and answers exceed batch limit.")
        return await asyncio.get_running_loop().run_in_executor(
            app.state.executor, app.state.engine.rank, request
        )

    @app.post("/v1/embeddings")
    async def embed(request: EmbeddingRequest) -> EmbeddingResponse:
        """Return one normalized 4096-dimensional vector per input text."""
        validate_model(request.model)
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
            app.state.executor, app.state.engine.encoder.embed,
            texts, max_tokens,
        )
        if request.encoding_format == "base64":
            vectors = [
                base64.b64encode(
                    struct.pack(f"<{len(vector)}f", *vector)
                ).decode("ascii")
                for vector in vectors
            ]
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
