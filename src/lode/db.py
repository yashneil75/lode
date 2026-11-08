"""SQLite layout for a lode index.

One index is one file. That makes delete and rename a filesystem move, lets an index be
copied between machines, and keeps the tool free of any server or daemon.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Iterator

from . import paths

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS file (
  id     INTEGER PRIMARY KEY,
  path   TEXT NOT NULL UNIQUE,
  sha256 TEXT NOT NULL,
  bytes  INTEGER NOT NULL,
  mtime  REAL NOT NULL,
  lang   TEXT NOT NULL,
  chunks INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS chunk (
  id         INTEGER PRIMARY KEY,
  file_id    INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
  ord        INTEGER NOT NULL,
  start_line INTEGER NOT NULL,
  end_line   INTEGER NOT NULL,
  symbol     TEXT NOT NULL DEFAULT '',
  kind       TEXT NOT NULL DEFAULT '',
  text       TEXT NOT NULL,
  vec        BLOB,
  dl         INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS chunk_by_file ON chunk(file_id);

CREATE TABLE IF NOT EXISTS posting (
  term     TEXT NOT NULL,
  chunk_id INTEGER NOT NULL,
  tf       INTEGER NOT NULL,
  PRIMARY KEY (term, chunk_id)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS posting_by_chunk ON posting(chunk_id);

CREATE TABLE IF NOT EXISTS term_stat (
  term TEXT PRIMARY KEY,
  df   INTEGER NOT NULL
) WITHOUT ROWID;
"""

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class BadName(ValueError):
    pass


def check_name(name: str) -> str:
    """Index names become filenames, so reject anything that could escape the store."""
    if not _NAME_RE.match(name or ""):
        raise BadName(
            f"invalid index name {name!r}: use 1-64 letters, digits, dot, dash or underscore"
        )
    return name


def exists(name: str) -> bool:
    return paths.index_path(check_name(name)).is_file()


def all_names() -> list[str]:
    directory = paths.indexes_dir()
    if not directory.is_dir():
        return []
    return sorted(p.name[: -len(".lode")] for p in directory.glob("*.lode"))


class Store:
    """A handle on one index file."""

    def __init__(self, name: str, *, readonly: bool = False) -> None:
        self.name = check_name(name)
        self.path = paths.index_path(self.name)
        self._open = True
        if not self.path.is_file() and readonly:
            raise FileNotFoundError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        if not readonly:
            self.conn.executescript(SCHEMA)
            self._check_version()
        self.meta = self._read_meta()

    def _check_version(self) -> None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'schema'").fetchone()
        current = int(row["value"]) if row else 0
        if current == 0:
            self.conn.execute("INSERT INTO meta VALUES ('schema', ?)", (str(SCHEMA_VERSION),))
        elif current > SCHEMA_VERSION:
            raise RuntimeError(
                f"index {self.name!r} uses schema v{current}; this lode understands v{SCHEMA_VERSION}"
            )

    def _read_meta(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in self.conn.execute("SELECT key, value FROM meta")}

    def set_meta(self, **items: object) -> None:
        for key, value in items.items():
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )
        self.meta.update({k: str(v) for k, v in items.items()})

    def touch_updated(self) -> None:
        self.set_meta(updated_at=int(time.time()))

    def commit(self) -> None:
        self.conn.commit()

    # -- files ---------------------------------------------------------------

    def file_row(self, relpath: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM file WHERE path = ?", (relpath,)).fetchone()

    def known_digests(self) -> dict[str, str]:
        """path -> sha256, so an update can skip files that did not change."""
        return {r["path"]: r["sha256"] for r in self.conn.execute("SELECT path, sha256 FROM file")}

    def file_paths(self) -> set[str]:
        return {r["path"] for r in self.conn.execute("SELECT path FROM file")}

    def write_file(
        self,
        relpath: str,
        digest: str,
        size: int,
        mtime: float,
        lang: str,
        units: list[tuple[int, int, int, str, str, str]],
        vectors: list[bytes | None],
        lexicons: list[dict[str, int]],
    ) -> int:
        """Replace everything stored for one path with the freshly computed chunks."""
        cur = self.conn
        previous = cur.execute("SELECT id FROM file WHERE path = ?", (relpath,)).fetchone()
        if previous:
            ids = [
                r["id"]
                for r in cur.execute("SELECT id FROM chunk WHERE file_id = ?", (previous["id"],))
            ]
            self.drop_chunks(ids)
            cur.execute("DELETE FROM file WHERE id = ?", (previous["id"],))

        file_id = cur.execute(
            "INSERT INTO file(path, sha256, bytes, mtime, lang, chunks) VALUES (?,?,?,?,?,?)",
            (relpath, digest, size, mtime, lang, len(units)),
        ).lastrowid

        postings: list[tuple[str, int, int]] = []
        df: dict[str, int] = {}
        for unit, vec, lex in zip(units, vectors, lexicons, strict=True):
            ord_, start, end, symbol, kind, text = unit
            chunk_id = cur.execute(
                "INSERT INTO chunk(file_id, ord, start_line, end_line, symbol, kind, text, vec, dl)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (file_id, ord_, start, end, symbol, kind, text, vec, sum(lex.values())),
            ).lastrowid
            for term, tf in lex.items():
                postings.append((term, chunk_id, tf))
                df[term] = df.get(term, 0) + 1

        cur.executemany("INSERT INTO posting(term, chunk_id, tf) VALUES (?,?,?)", postings)
        cur.executemany(
            "INSERT INTO term_stat(term, df) VALUES (?, ?) "
            "ON CONFLICT(term) DO UPDATE SET df = df + excluded.df",
            list(df.items()),
        )
        return len(units)

    def drop_chunks(self, chunk_ids: list[int]) -> None:
        if not chunk_ids:
            return
        cur = self.conn
        marks = ",".join("?" * len(chunk_ids))
        # df is the number of chunks containing a term, so removing postings changes it.
        counts = cur.execute(
            f"SELECT term, COUNT(*) AS n FROM posting WHERE chunk_id IN ({marks}) GROUP BY term",
            chunk_ids,
        ).fetchall()
        cur.execute(f"DELETE FROM posting WHERE chunk_id IN ({marks})", chunk_ids)
        for row in counts:
            cur.execute("UPDATE term_stat SET df = df - ? WHERE term = ?", (row["n"], row["term"]))
        cur.execute("DELETE FROM term_stat WHERE df <= 0")
        cur.execute(f"DELETE FROM chunk WHERE id IN ({marks})", chunk_ids)

    def remove_file(self, relpath: str) -> int:
        row = self.conn.execute("SELECT id FROM file WHERE path = ?", (relpath,)).fetchone()
        if not row:
            return 0
        ids = [
            r["id"] for r in self.conn.execute("SELECT id FROM chunk WHERE file_id = ?", (row["id"],))
        ]
        self.drop_chunks(ids)
        self.conn.execute("DELETE FROM file WHERE id = ?", (row["id"],))
        return len(ids)

    def prune_files(self, keep: set[str]) -> list[str]:
        """Forget paths that disappeared from disk."""
        gone = [p for p in self.file_paths() if p not in keep]
        for path in gone:
            self.remove_file(path)
        return gone

    def reclaim(self) -> None:
        """Compact the file after bulk deletions."""
        self.conn.commit()
        self.conn.execute("VACUUM")

    # -- chunk reads ---------------------------------------------------------

    def vector_rows(self) -> Iterator[tuple[int, bytes]]:
        """Keyset pagination: OFFSET would rescan for every batch on large indexes."""
        last = -1
        while True:
            rows = self.conn.execute(
                "SELECT id, vec FROM chunk WHERE vec IS NOT NULL AND id > ? ORDER BY id LIMIT 8192",
                (last,),
            ).fetchall()
            if not rows:
                return
            for row in rows:
                yield row["id"], row["vec"]
            last = rows[-1]["id"]

    def chunk_row(self, chunk_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT c.*, f.path, f.lang FROM chunk c JOIN file f ON f.id = c.file_id"
            " WHERE c.id = ?",
            (chunk_id,),
        ).fetchone()

    def counts(self) -> dict[str, float]:
        one = lambda sql: self.conn.execute(sql).fetchone()  # noqa: E731
        files, nbytes = one("SELECT COUNT(*), COALESCE(SUM(bytes),0) FROM file")
        chunks, total_dl = one("SELECT COUNT(*), COALESCE(SUM(dl),0) FROM chunk")
        embedded = one("SELECT COUNT(*) FROM chunk WHERE vec IS NOT NULL")[0]
        terms = one("SELECT COUNT(*) FROM term_stat")[0]
        return {
            "files": files,
            "bytes": nbytes,
            "chunks": chunks,
            "avg_dl": (total_dl / chunks) if chunks else 0.0,
            "terms": terms,
            "embedded": embedded,
        }

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        if self._open:
            self.conn.close()
        self._open = False

    def _sidecars(self) -> list[Path]:
        directory = paths.indexes_dir()
        stem = self.name + ".lode"
        return [directory / f"{stem}-wal", directory / f"{stem}-shm", paths.index_path(self.name)]

    def destroy(self) -> None:
        self.close()
        for candidate in self._sidecars():
            candidate.unlink(missing_ok=True)

    def rename(self, new_name: str) -> Path:
        check_name(new_name)
        if paths.index_path(new_name).exists():
            raise FileExistsError(f"an index named {new_name!r} already exists")
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.close()
        for source in self._sidecars():
            if not source.exists():
                continue
            suffix = source.name[len(self.name):]
            source.rename(paths.indexes_dir() / (new_name + suffix))
        self.name = new_name
        self.path = paths.index_path(new_name)
        return self.path

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        if self._open:
            self.conn.commit()
            self.close()
