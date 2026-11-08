"""Indexing: coverage, incrementality, and degrading to lexical when the model fails."""

from __future__ import annotations

import pytest

from lode import db, index as indexer


@pytest.fixture
def store(tmp_path):
    with db.Store("index-test") as handle:
        yield handle


def relpaths(store):
    return {row["path"] for row in store.conn.execute("SELECT path FROM file")}


def test_walk_finds_sources_and_skips_noise(sample_project):
    ignores = indexer.Ignores(sample_project)
    found = {
        indexer.relative(path, sample_project)
        for path in indexer.walk(sample_project, ignores)
    }
    assert "src/auth.py" in found
    assert "README.md" in found
    assert "node_modules/dep.js" not in found, "vendored code is not your code"


def test_binary_files_are_detected_not_garbled(sample_project):
    content = indexer.read_source(sample_project / "asset.bin", 1_000)
    assert content is None


def test_oversize_files_are_skipped(sample_project, store, embedder_):
    big = sample_project / "src" / "generated.py"
    big.write_text("x = 1\n" * 100_000, encoding="utf-8")
    report = indexer.build(store, sample_project, max_bytes=4096, provider=embedder_)
    assert report.skipped_large >= 1
    assert "src/generated.py" not in relpaths(store)


def test_indexing_covers_the_project(store, sample_project, embedder_):
    report = indexer.build(store, sample_project, provider=embedder_)
    assert relpaths(store) == {"src/auth.py", "src/db.py", "src/login.js", "README.md"}
    assert report.chunks == store.counts()["chunks"]
    assert store.meta["root"] == str(sample_project.resolve())
    assert store.meta["model"] == embedder_.alias


def test_every_stored_chunk_has_a_vector(store, sample_project, embedder_):
    indexer.build(store, sample_project, provider=embedder_)
    counts = store.counts()
    assert counts["embedded"] == counts["chunks"]


def test_second_run_is_a_no_op(store, sample_project, embedder_):
    indexer.build(store, sample_project, provider=embedder_)
    before = embedder_.embedded
    report = indexer.build(store, sample_project, provider=embedder_)
    assert report.touched == 0
    assert report.unchanged == report.files
    assert embedder_.embedded == before, "unchanged files must not be re-embedded"


def test_only_the_changed_file_is_rewritten(store, sample_project, embedder_):
    indexer.build(store, sample_project, provider=embedder_)
    before = embedder_.embedded
    (sample_project / "src" / "db.py").write_text("def migrate():\n    return 1\n", encoding="utf-8")
    report = indexer.build(store, sample_project, provider=embedder_)
    assert report.added == 0 and report.updated == 1
    assert embedder_.embedded - before <= 2, "one file changed, so one file was embedded"


def test_new_files_are_added_and_deleted_ones_pruned(store, sample_project, embedder_):
    indexer.build(store, sample_project, provider=embedder_)
    (sample_project / "src" / "cache.py").write_text("def get(key):\n    return key\n", encoding="utf-8")
    (sample_project / "src" / "db.py").unlink()
    report = indexer.build(store, sample_project, provider=embedder_)
    assert report.added == 1
    paths = relpaths(store)
    assert "src/cache.py" in paths and "src/db.py" not in paths


def test_rebuild_replaces_everything(store, sample_project, embedder_):
    indexer.build(store, sample_project, provider=embedder_)
    report = indexer.build(store, sample_project, provider=embedder_, rebuild=True)
    assert report.added == report.files


def test_keyword_only_when_the_model_is_unavailable(store, sample_project):
    report = indexer.build(store, sample_project, provider=DeadEmbedder())
    assert report.warnings, "the user must be told embeddings did not happen"
    assert report.chunks > 0, "the lexical half still has to exist"
    assert store.counts()["embedded"] == 0


def test_no_embeddings_option_leaves_vectors_empty(store, sample_project, embedder_):
    indexer.build(store, sample_project, provider=embedder_, use_embeddings=False)
    assert store.counts()["embedded"] == 0
    assert store.counts()["chunks"] > 0


def test_extra_ignore_patterns_are_honoured(store, sample_project, embedder_):
    indexer.build(store, sample_project, provider=embedder_, extra_ignore=["src/db.py"])
    assert "src/db.py" not in relpaths(store)


def test_nested_gitignore_applies(store, sample_project, embedder_):
    (sample_project / "src" / ".gitignore").write_text("db.py\n", encoding="utf-8")
    indexer.build(store, sample_project, provider=embedder_)
    assert "src/db.py" not in relpaths(store)


def test_extension_filter_reduces_the_corpus(store, sample_project, embedder_):
    indexer.build(store, sample_project, provider=embedder_, extensions=(".py",))
    paths = relpaths(store)
    assert paths and all(path.endswith(".py") for path in paths)


def test_empty_directory_produces_an_empty_index(store, tmp_path, embedder_):
    empty = tmp_path / "empty"
    empty.mkdir()
    report = indexer.build(store, empty, provider=embedder_)
    assert report.files == 0
    assert store.counts()["chunks"] == 0


def test_to_blob_roundtrip():
    import numpy as np

    vector = np.linspace(-1, 1, 16, dtype=np.float32)
    restored = np.frombuffer(indexer.to_blob(vector), dtype=np.float32)
    assert np.allclose(restored, vector)


class DeadEmbedder:
    alias = "fake/dead"
    dim = None
    error = None

    @property
    def usable(self):
        return False

    def load(self, *, quiet: bool = False):
        self.error = "weights could not be fetched"

    def prepare(self, *, quiet: bool = False, force: bool = False):
        raise RuntimeError("weights could not be fetched")

    def embed(self, texts, batch_size: int = 64):
        raise AssertionError("indexer embedded with an unusable model")
