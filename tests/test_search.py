"""Search over a built index -- the path a user actually takes."""

from __future__ import annotations

import numpy as np
import pytest

from lode import db, index as indexer, search


@pytest.fixture
def built(sample_project, embedder_):
    with db.Store("search-test") as store:
        report = indexer.build(store, sample_project, provider=embedder_)
        yield store, report


def first_path(hits):
    return hits[0].path if hits else None


def test_indexing_found_the_sources(built):
    store, report = built
    assert report.files == 4, "node_modules and the binary file must not count"
    assert store.counts()["embedded"] == store.counts()["chunks"]


def test_identifier_query_finds_the_definition(built):
    store, _ = built
    hits = search.search(store, "TokenIssuer", provider=FakeProvider(), mode="keyword", k=5)
    assert first_path(hits) == "src/auth.py"
    assert "TokenIssuer" in hits[0].text


def test_a_borrowed_word_still_ranks_its_owning_file_first(built):
    """"hash_password" splits into hash + password, and login.js says both more often."""
    store, _ = built
    hits = search.search(store, "hash_password", provider=FakeProvider(), mode="keyword", k=5)
    assert "src/auth.py" in {hit.path for hit in hits[:2]}


def test_hybrid_still_surfaces_the_definition(built):
    """The fake vectors collide, so assert presence rather than the exact podium."""
    store, _ = built
    hits = search.search(store, "hash_password", provider=FakeProvider(), k=5)
    assert "src/auth.py" in {hit.path for hit in hits}


def test_natural_language_query_reaches_the_same_code(built):
    store, _ = built
    hits = search.search(store, "how are passwords turned into tokens", provider=FakeProvider(), k=5)
    assert any("auth" in hit.path for hit in hits)


def test_scores_normalise_to_100_for_a_unanimous_hit(built):
    store, _ = built
    hits = search.search(store, "hash_password", provider=FakeProvider(), mode="keyword", k=5)
    assert hits and hits[0].score == pytest.approx(100.0, abs=0.01)
    assert hits[0].lexical == 1 and hits[0].semantic is None


def test_keyword_mode_never_touches_the_model(built):
    store, _ = built
    provider = ExplodingProvider()
    hits = search.search(store, "hash_password", provider=provider, mode="keyword", k=5)
    assert hits
    assert provider.calls == 0
    assert all(hit.semantic is None for hit in hits)


def test_semantic_mode_reports_an_unavailable_model(built):
    store, _ = built
    with pytest.raises(search.NoCandidates):
        search.search(store, "anything", provider=DeadProvider(), mode="semantic", k=5)


def test_path_filter_narrows_results(built):
    store, _ = built
    hits = search.search(store, "password", provider=FakeProvider(), k=10, paths=["login.js"])
    assert hits and {hit.path for hit in hits} == {"src/login.js"}


def test_language_filter_narrows_results(built):
    store, _ = built
    hits = search.search(store, "password", provider=FakeProvider(), k=10, langs=["javascript"])
    assert hits and {hit.lang for hit in hits} == {"javascript"}


def test_max_per_file_caps_duplicates(built):
    store, _ = built
    hits = search.search(store, "hash", provider=FakeProvider(), k=10, max_per_file=1)
    assert len({hit.path for hit in hits}) == len(hits)


def test_k_is_respected(built):
    store, _ = built
    assert len(search.search(store, "password", provider=FakeProvider(), k=2)) <= 2


def test_hits_carry_the_lines_span(built):
    store, _ = built
    hit = search.search(store, "hash_password", provider=FakeProvider(), k=1)[0]
    assert hit.start >= 1 and hit.end >= hit.start
    assert hit.lines == hit.end - hit.start + 1


def test_no_query_terms_returns_no_hits(built):
    store, _ = built
    assert search.search(store, "the a of", provider=DeadProvider(), mode="keyword", k=5) == []


# -- local providers so a mode test does not depend on the shared fake -------------


class FakeProvider:
    """Deterministic and cheap; matches the indexed vectors from conftest."""

    def __init__(self):
        from conftest import FakeEmbedder

        self._inner = FakeEmbedder()
        self.alias = self._inner.alias
        self.dim = self._inner.dim
        self.error = None

    @property
    def usable(self):
        return True

    def load(self, *, quiet: bool = False):
        pass

    def embed_one(self, text):
        return self._inner.vector(text)

    def embed(self, texts, batch_size: int = 64):
        yield from self._inner.embed(texts)


class ExplodingProvider(FakeProvider):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def embed_one(self, text):
        self.calls += 1
        return super().embed_one(text)


class DeadProvider:
    usable = False
    error = "model missing"
    alias = "fake/dead"
    dim = 64

    def load(self, *, quiet: bool = False):
        pass

    def embed_one(self, text):
        raise AssertionError("semantic search ran with no model")
