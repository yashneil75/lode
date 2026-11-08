"""BM25 over the inverted index stored beside the vectors."""

from __future__ import annotations

import math

import pytest

from lode import bm25, db, lexicon


@pytest.fixture
def store(tmp_path):
    with db.Store("bm25-test") as handle:
        yield handle


def index_chunks(store, chunks: list[tuple[str, str]]) -> dict[str, int]:
    """(path, text) in, {path: chunk_id} out -- one chunk per file keeps ids predictable."""
    ids: dict[str, int] = {}
    for path, text in chunks:
        lex = lexicon.lexicon(path, "", "", text)
        store.write_file(
            path, f"digest-{path}", len(text), 1.0, "python",
            [(0, 1, 10, "", "def", text)], [None], [dict(lex)],
        )
        ids[path] = store.conn.execute(
            "SELECT c.id AS id FROM chunk c JOIN file f ON f.id = c.file_id WHERE f.path = ?",
            (path,),
        ).fetchone()["id"]
    return ids


def filler(count: int, start: int = 0) -> str:
    return " ".join(f"fillerword{i}" for i in range(start, start + count))


def corpus(store, named: dict[str, str]) -> dict[str, int]:
    """A small corpus, where every term stays rankable because nothing is common yet."""
    chunks = list(named.items())
    for index in range(8):
        chunks.append((f"pad{index}.py", filler(12, start=index * 12)))
    return index_chunks(store, chunks)


def large_corpus(store, named: dict[str, str], pads: int = 30) -> dict[str, int]:
    """Big enough that the common-term cut-in actually engages."""
    chunks = list(named.items())
    for index in range(pads):
        chunks.append((f"pad{index}.py", filler(12, start=index * 12)))
    return index_chunks(store, chunks)


def test_returns_nothing_for_an_empty_index(store):
    assert bm25.rank(store, ["anything"]) == []


def test_returns_nothing_when_the_query_has_no_rankable_terms(store):
    index_chunks(store, [("a.py", "alpha beta")])
    assert bm25.rank(store, lexicon.query_terms("the a of")) == []


def test_a_rare_term_finds_its_document(store):
    ids = corpus(store, {"needle.py": f"needle {filler(10)}"})
    ranked = bm25.rank(store, ["needle"])
    assert ranked and ranked[0][0] == ids["needle.py"]


def test_shorter_document_wins_the_same_term(store):
    short = f"needle {filler(4)}"
    long = f"needle {filler(60)}"
    ids = corpus(store, {"short.py": short, "long.py": long})
    scores = dict(bm25.rank(store, ["needle"]))
    assert scores[ids["short.py"]] > scores[ids["long.py"]]


def test_more_matching_terms_outrank_a_higher_term_frequency(store):
    both = f"needle sprocket {filler(4, start=200)}"
    repeated = " ".join(["needle"] * 6) + " " + filler(4, start=200)
    ids = corpus(store, {"both.py": both, "repeated.py": repeated})
    scores = dict(bm25.rank(store, ["needle", "sprocket"]))
    assert scores[ids["both.py"]] > scores[ids["repeated.py"]]


def test_require_all_drops_partial_matches(store):
    ids = corpus(store, {"a.py": f"needle {filler(4)}", "b.py": f"sprocket {filler(4)}"})
    both = bm25.rank(store, ["needle", "sprocket"])
    assert {chunk_id for chunk_id, _ in both} == {ids["a.py"], ids["b.py"]}
    strict = bm25.rank(store, ["needle", "sprocket"], require_all=True)
    assert strict == []


def test_require_all_keeps_a_chunk_holding_every_term(store):
    ids = corpus(store, {"a.py": f"needle sprocket {filler(4)}"})
    strict = bm25.rank(store, ["needle", "sprocket"], require_all=True)
    assert [chunk_id for chunk_id, _ in strict] == [ids["a.py"]]


def test_allow_set_restricts_candidates(store):
    ids = corpus(store, {"a.py": f"needle {filler(4)}", "b.py": f"needle {filler(4)}"})
    ranked = bm25.rank(store, ["needle"], allow={ids["a.py"]})
    assert [chunk_id for chunk_id, _ in ranked] == [ids["a.py"]]


def test_a_term_in_every_chunk_is_dropped_once_the_corpus_is_large(store):
    chunks = {f"f{i}.py": f"common {filler(6, start=i * 6)}" for i in range(30)}
    chunks["special.py"] = "common needle " + filler(6, start=900)
    ids = index_chunks(store, list(chunks.items()))
    ranked = bm25.rank(store, ["needle", "common"])
    assert [chunk_id for chunk_id, _ in ranked] == [ids["special.py"]]


def test_a_single_term_query_still_uses_an_otherwise_common_term(store):
    chunks = {f"f{i}.py": f"common {filler(6, start=i * 6)}" for i in range(30)}
    ids = index_chunks(store, list(chunks.items()))
    assert len(bm25.rank(store, ["common"])) == len(ids)


def test_a_small_corpus_never_drops_a_term(store):
    """In a folder of a dozen chunks every identifier clears 25%; skipping them all
    would silently turn search off."""
    chunks = {f"f{i}.py": f"common needle{i} {filler(4, start=i * 4)}" for i in range(6)}
    ids = index_chunks(store, list(chunks.items()))
    ranked = bm25.rank(store, ["needle", "common"])
    assert len(ranked) == len(ids)


def test_term_frequency_saturates(store):
    once = f"needle {filler(6)}"
    many = " ".join(["needle"] * 40) + " " + filler(6)
    ids = corpus(store, {"once.py": once, "many.py": many})
    scores = dict(bm25.rank(store, ["needle"]))
    ratio = scores[ids["many.py"]] / scores[ids["once.py"]]
    assert 1.0 < ratio < 3.0, "tf saturation: 40 hits must not beat 1 hit by 40x"


def test_limit_is_respected(store):
    index_chunks(store, [(f"f{i}.py", f"needle {filler(6, start=i * 6)}") for i in range(10)])
    assert len(bm25.rank(store, ["needle"], limit=3)) == 3


def test_scores_are_positive_and_ordered(store):
    corpus(store, {"a.py": f"needle {filler(4)}", "b.py": f"sprocket {filler(4)}"})
    scores = [score for _, score in bm25.rank(store, ["needle", "sprocket"])]
    assert scores == sorted(scores, reverse=True)
    assert all(score > 0 for score in scores)


def test_idf_is_monotonic_in_rarity(store):
    assert bm25._idf(1, 100) > bm25._idf(50, 100) > bm25._idf(99, 100)
    assert math.isfinite(bm25._idf(100, 100))
