"""Shared fixtures. Nothing here may touch the network or a real ONNX model."""

from __future__ import annotations

import hashlib
from typing import Iterable, Iterator

import numpy as np
import pytest

from lode import embedder, lexicon, paths

DIM = 64


class FakeEmbedder:
    """Bag-of-terms hashed into a unit vector: same shape as Embedder, no model.

    Cosine over these tracks term overlap, so semantic ranking agrees with BM25 on
    identifier-shaped queries -- which is what the fusion tests need to assert on.
    """

    def __init__(self, alias: str = "fake/test-model", dim: int = DIM) -> None:
        self.alias = alias
        self.dim = dim
        self.error: str | None = None
        self.embedded = 0
        self.load_calls = 0

    @property
    def usable(self) -> bool:
        return True

    def load(self, *, quiet: bool = False) -> None:
        self.load_calls += 1

    def prepare(self, *, quiet: bool = False, force: bool = False):
        return embedder.ModelInfo(alias=self.alias, dim=self.dim, max_tokens=512, mb=0.1)

    def vector(self, text: str) -> np.ndarray:
        row = np.zeros(self.dim, dtype=np.float32)
        for term, count in lexicon.counts(text).items():
            slot = int.from_bytes(hashlib.blake2b(term.encode(), digest_size=4).digest(), "big")
            row[slot % self.dim] += float(count)
        norm = float(np.linalg.norm(row))
        return row / norm if norm else row

    def embed(self, texts: Iterable[str], batch_size: int = 64) -> Iterator[np.ndarray]:
        for text in texts:
            self.embedded += 1
            yield self.vector(text)

    def embed_one(self, text: str) -> np.ndarray:
        return self.vector(text)


class DeadEmbedder(FakeEmbedder):
    """Stands in for a machine that cannot run the model: index must stay lexical."""

    @property
    def usable(self) -> bool:
        return False

    def load(self, *, quiet: bool = False) -> None:
        self.error = "onnxruntime is not installed"

    def embed(self, texts, batch_size: int = 64):  # pragma: no cover - must never run
        raise AssertionError("indexer embedded with an unusable model")

    def embed_one(self, text: str):  # pragma: no cover - must never run
        raise AssertionError("query embedded with an unusable model")


@pytest.fixture(autouse=True)
def lode_home(tmp_path, monkeypatch):
    """Every test gets its own index store and cannot download anything."""
    home = tmp_path / "lode-home"
    monkeypatch.setenv("LODE_HOME", str(home))
    monkeypatch.delenv("LODE_MODEL", raising=False)
    monkeypatch.delenv("LODE_NO_EMBED", raising=False)
    monkeypatch.setattr(paths, "model_cache_dir", lambda: home / "models")
    return home


@pytest.fixture
def embedder_(tmp_path):
    return FakeEmbedder()


@pytest.fixture
def sample_project(tmp_path):
    """A five-file repo covering python, js, markdown and the ignore rules."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "auth.py").write_text(
        '"""Login and token handling."""\n'
        "import hashlib\n\n\n"
        "def hash_password(password: str) -> str:\n"
        "    return hashlib.sha256(password.encode()).hexdigest()\n\n\n"
        "class TokenIssuer:\n"
        "    def issue(self, user_id: int) -> str:\n"
        '        return hash_password(f"token-{user_id}")\n',
        encoding="utf-8",
    )
    (root / "src" / "db.py").write_text(
        "def connect(database: str):\n"
        '    print("opening", database)\n\n\n'
        "def run_query(sql: str):\n"
        "    return sql.strip()\n",
        encoding="utf-8",
    )
    (root / "src" / "login.js").write_text(
        "export function checkPassword(user, password) {\n"
        "  if (!user || !password) {\n"
        "    throw new Error('missing credentials');\n"
        "  }\n"
        "  return user.passwordHash === hash(password);\n"
        "}\n\n"
        "function hash(value) {\n"
        "  return value.length;\n"
        "}\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# Repo\n\n## Authentication\n\nPasswords are hashed before storage.\n",
        encoding="utf-8",
    )
    (root / "node_modules").mkdir()
    (root / "node_modules" / "dep.js").write_text("export const noise = 1;\n", encoding="utf-8")
    (root / "asset.bin").write_bytes(b"\x00\x01\x02binary\xff")
    return root
