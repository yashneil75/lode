"""Embedding provider: ONNX bge-small via fastembed, with a keyword-only escape hatch."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

from . import modelsource, paths

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
BATCH = 64

_MODEL_ENV = "LODE_MODEL"
_OFFLINE_ENV = "LODE_NO_EMBED"


@dataclass(frozen=True)
class ModelInfo:
    alias: str
    dim: int
    max_tokens: int
    mb: float


def preferred_model() -> str:
    from . import config

    return os.environ.get(_MODEL_ENV) or config.get("default_model") or DEFAULT_MODEL


def is_disabled() -> bool:
    return os.environ.get(_OFFLINE_ENV, "").strip().upper() in {"1", "TRUE", "YES", "ON"}


class Embedder:
    """Lazily-loaded embedding model.

    A failure to obtain weights is not fatal: callers check `usable` and fall back to
    lexical ranking, which still returns useful hits for identifier-like queries.
    """

    def __init__(self, alias: str | None = None, threads: int | None = None) -> None:
        wanted = alias or preferred_model()
        self.alias = modelsource.canonical(wanted)
        self.threads = threads or max(1, min(8, os.cpu_count() or 2))
        self._model = None
        self._dir = None
        self.error: str | None = None
        self.dim: int | None = None

    @property
    def usable(self) -> bool:
        return self._model is not None

    def prepare(self, *, quiet: bool = False, force: bool = False) -> ModelInfo:
        """Fetch weights without starting an ONNX session."""
        spec = modelsource.resolve_spec(self.alias)
        directory = modelsource.ensure(self.alias, quiet=quiet, force=force)
        return ModelInfo(
            alias=spec.alias,
            dim=spec.dim,
            max_tokens=spec.max_tokens,
            mb=_size_mb(directory),
        )

    def load(self, *, quiet: bool = False) -> None:
        if self._model is not None:
            return
        if is_disabled():
            self.error = f"{_OFFLINE_ENV} is set, running keyword-only"
            return
        from fastembed import TextEmbedding

        try:
            self._dir = modelsource.ensure(self.alias, quiet=quiet)
            self._model = TextEmbedding(
                model_name=self.alias,
                specific_model_path=str(self._dir),
                threads=self.threads,
                lazy_load=False,
                cache_dir=str(paths.model_cache_dir()),
            )
            spec = modelsource.resolve_spec(self.alias)
            self.dim = spec.dim
        except modelsource.ModelUnavailable as exc:
            self.error = str(exc)
        except Exception as exc:  # onnxruntime init, missing files, bad session
            self.error = f"{type(exc).__name__}: {exc}"

    def embed(self, texts: Iterable[str], batch_size: int = BATCH) -> Iterator[np.ndarray]:
        if self._model is None:
            self.load()
        if self._model is None:
            raise RuntimeError(self.error or "embedding model unavailable")
        buffer: list[str] = []
        for text in texts:
            buffer.append(_truncate(text))
            if len(buffer) >= batch_size:
                yield from self._run(buffer)
                buffer = []
        if buffer:
            yield from self._run(buffer)

    def _run(self, batch: list[str]) -> list[np.ndarray]:
        out = [np.asarray(v, dtype=np.float32) for v in self._model.embed(batch)]  # type: ignore[union-attr]
        return [_l2(v) for v in out]

    def embed_one(self, text: str) -> np.ndarray:
        return next(iter(self.embed([text])))

    def close(self) -> None:
        self._model = None


def stack(vectors: Iterable[np.ndarray]) -> np.ndarray:
    rows = list(vectors)
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(rows).astype(np.float32, copy=False)


def _l2(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def _truncate(text: str) -> str:
    # Cheap guard against pathological inputs: ~4 chars per token is a safe floor for
    # byte-pair encodings, and the model truncates beyond its window anyway.
    limit = 512 * 4
    return text[:limit] if len(text) > limit else text


def _size_mb(directory) -> float:
    total = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    return round(total / 1e6, 1)
