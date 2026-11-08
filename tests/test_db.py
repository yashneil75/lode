"""The store is the only thing with durable state, so its invariants are load-bearing."""

from __future__ import annotations

import numpy as np
import pytest

from lode import db, index as indexer, lexicon


@pytest.fixture
def store(tmp_path):
    with db.Store("unit-test") as handle:
        yield handle


def add_file(store, path, texts, vectors=None, lexicons=None):
    units = [(i, 1, 2, f"s{i}", "def", text) for i, text in enumerate(texts)]
    vectors = vectors if vectors is not None else [None] * len(texts)
    lexicons = (
        lexicons
        if lexicons is not None
        else [dict(lexicon.lexicon(path, "", "", text)) for text in texts]
    )
    return store.write_file(path, f"digest-{path}", 10, 1.0, "python", units, vectors, lexicons)


def df_of(store, term):
    row = store.conn.execute("SELECT df FROM term_stat WHERE term = ?", (term,)).fetchone()
    return None if row is None else row["df"]


def test_name_validation_blocks_path_traversal():
    assert db.check_name("my-index.v2") == "my-index.v2"
    for bad in ("", "../escape", "with space", "-leading-dash", "x" * 65, "nested/name"):
        with pytest.raises(db.BadName):
            db.check_name(bad)


def test_write_file_returns_the_chunk_count(store):
    assert add_file(store, "a.py", ["alpha first", "alpha second"]) == 2
    assert store.counts()["chunks"] == 2


def test_rewriting_a_path_replaces_its_chunks(store):
    add_file(store, "a.py", ["one thing", "two thing", "three thing"])
    add_file(store, "a.py", ["only one"])
    counts = store.counts()
    assert counts["chunks"] == 1
    assert counts["files"] == 1


def test_df_counts_documents_not_postings(store):
    add_file(store, "a.py", ["shared term here"])
    add_file(store, "b.py", ["shared other thing"])
    assert df_of(store, "shared") == 2


def test_repeating_a_term_in_one_chunk_does_not_inflate_df(store):
    add_file(store, "a.py", ["shared shared shared"])
    assert df_of(store, "shared") == 1


def test_drop_chunks_decrements_df(store):
    add_file(store, "a.py", ["shared one"])
    add_file(store, "b.py", ["shared two"])
    ids = [
        r["id"]
        for r in store.conn.execute(
            "SELECT id FROM chunk WHERE file_id = (SELECT id FROM file WHERE path = 'a.py')"
        )
    ]
    store.drop_chunks(ids)
    assert df_of(store, "shared") == 1


def test_df_never_goes_negative(store):
    add_file(store, "a.py", ["lonely term"])
    store.drop_chunks([r["id"] for r in store.conn.execute("SELECT id FROM chunk")])
    assert df_of(store, "lonely") is None
    assert store.conn.execute("SELECT COUNT(*) FROM term_stat").fetchone()[0] == 0


def test_prune_files_forgets_deleted_paths(store):
    add_file(store, "keep.py", ["keep me"])
    add_file(store, "gone.py", ["gone me"])
    assert store.prune_files({"keep.py"}) == ["gone.py"]
    assert store.file_paths() == {"keep.py"}
    assert store.counts()["chunks"] == 1


def test_remove_file_reports_dropped_chunks(store):
    add_file(store, "a.py", ["one thing", "two thing"])
    assert store.remove_file("a.py") == 2
    assert store.remove_file("a.py") == 0


def test_vector_rows_are_keyset_paginated(store):
    vectors = [indexer.to_blob(np.ones(4, dtype=np.float32) * i) for i in range(3)]
    add_file(store, "a.py", ["one", "two", "three"], vectors=vectors)
    rows = list(store.vector_rows())
    assert len(rows) == 3
    assert [np.frombuffer(blob, dtype=np.float32)[0] for _, blob in rows] == [0.0, 1.0, 2.0]


def test_chunk_row_carries_path_and_language(store):
    add_file(store, "a.py", ["one thing"])
    chunk_id = store.conn.execute("SELECT id FROM chunk").fetchone()["id"]
    row = store.chunk_row(chunk_id)
    assert row["path"] == "a.py"
    assert row["lang"] == "python"
    assert store.chunk_row(10_000) is None


def test_meta_survives_a_reopen():
    with db.Store("durable") as handle:
        handle.set_meta(root="/tmp/x", dim="384")
        assert handle.meta["root"] == "/tmp/x"
    reopened = db.Store("durable")
    try:
        assert reopened.meta["root"] == "/tmp/x"
        assert reopened.meta["dim"] == "384"
    finally:
        reopened.close()


def test_counts_on_an_empty_index(store):
    counts = store.counts()
    assert counts["chunks"] == 0
    assert counts["terms"] == 0
    assert counts["avg_dl"] == 0.0


def test_counts_after_writing(store):
    add_file(store, "a.py", ["alpha one", "beta two"])
    counts = store.counts()
    assert counts == {
        "files": 1,
        "bytes": 10,
        "chunks": 2,
        "avg_dl": counts["avg_dl"],
        "terms": counts["terms"],
        "embedded": 0,
    }


def test_destroy_removes_the_file_and_sidecars():
    with db.Store("doomed") as handle:
        add_file(handle, "a.py", ["one"])
    assert db.exists("doomed")
    db.Store("doomed").destroy()
    assert not db.exists("doomed")
    assert not (db.paths.indexes_dir() / "doomed.lode-wal").exists()


def test_rename_moves_the_file_and_keeps_contents():
    with db.Store("before") as handle:
        add_file(handle, "a.py", ["one"])
    db.Store("before").rename("after")
    assert not db.exists("before")
    with db.Store("after") as handle:
        assert handle.counts()["chunks"] == 1
        assert handle.name == "after"


def test_rename_rejects_a_bad_target():
    with db.Store("ok") as handle:
        add_file(handle, "a.py", ["one"])
    with pytest.raises(db.BadName):
        db.Store("ok").rename("../escaped")
    assert db.exists("ok")


def test_renaming_onto_an_existing_index_is_refused():
    with db.Store("one") as handle:
        add_file(handle, "a.py", ["one"])
    with db.Store("two") as handle:
        add_file(handle, "b.py", ["two"])
    with pytest.raises(FileExistsError):
        db.Store("one").rename("two")
    assert db.exists("one") and db.exists("two")
