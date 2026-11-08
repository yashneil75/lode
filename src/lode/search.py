"""Hybrid retrieval: BM25 and cosine, fused with reciprocal rank fusion."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import bm25, embedder, lexicon
from .db import Store

BATCH_ROWS = 8192
RRF_K = 60.0
CANDIDATE_MULTIPLIER = 6
MAX_PER_FILE_DEFAULT = 3


class NoCandidates(Exception):
    pass


@dataclass
class Hit:
    chunk_id: int
    path: str
    start: int
    end: int
    symbol: str
    kind: str
    text: str
    score: float
    semantic: float | None
    lexical: float | None
    lang: str = ""
    root: str = ""
    ranks: tuple[int | None, int | None] = (None, None)

    @property
    def lines(self) -> int:
        return self.end - self.start + 1


def path_matches(relpath: str, patterns: Sequence[str]) -> bool:
    """A pattern without a slash matches the basename, like a shell `ls` would."""
    if not patterns:
        return True
    lowered = relpath.lower()
    base = lowered.rsplit("/", 1)[-1]
    for pattern in patterns:
        needle = pattern.lower().lstrip("./")
        if "/" in needle or "\\" in needle:
            if fnmatch.fnmatch(lowered, needle) or fnmatch.fnmatch(lowered, f"*/{needle}"):
                return True
        elif fnmatch.fnmatch(base, needle):
            return True
    return False


def vector_scores(
    store: Store,
    query: np.ndarray,
    *,
    paths: Sequence[str] = (),
    langs: Sequence[str] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity for every stored vector, in bounded memory.

    Vectors are already unit length, so the dot product is the cosine. Streaming in
    batches keeps a million-chunk index at a few tens of megabytes instead of gigabytes.
    """
    sql = (
        "SELECT c.id AS id, c.vec AS vec, f.path AS path, f.lang AS lang"
        " FROM chunk c JOIN file f ON f.id = c.file_id"
        " WHERE c.vec IS NOT NULL AND c.id > ? ORDER BY c.id LIMIT ?"
    )
    ids: list[int] = []
    sims: list[float] = []
    last = -1
    while True:
        rows = store.conn.execute(sql, (last, BATCH_ROWS)).fetchall()
        if not rows:
            break
        wanted = [
            row
            for row in rows
            if (not paths or path_matches(row["path"], paths))
            and (not langs or row["lang"] in langs)
        ]
        if wanted:
            block = np.frombuffer(
                b"".join(memoryview(row["vec"]) for row in wanted), dtype=np.float32
            ).reshape(len(wanted), -1)
            scores = block @ query
            ids.extend(row["id"] for row in wanted)
            sims.extend(np.asarray(scores, dtype=np.float32).tolist())
        last = rows[-1]["id"]
    return np.asarray(ids, dtype=np.int64), np.asarray(sims, dtype=np.float32)


def top_by_score(ids: np.ndarray, sims: np.ndarray, k: int) -> list[tuple[int, float]]:
    if ids.size == 0:
        return []
    order = np.argpartition(-sims, min(k, ids.size - 1))[:k] if ids.size > k else np.argsort(-sims)
    ranked = sorted(((int(ids[i]), float(sims[i])) for i in order), key=lambda x: (-x[1], x[0]))
    return ranked


def reciprocal_rank_fusion(
    lists: Sequence[Sequence[tuple[int, float]]],
    *,
    weights: Sequence[float] | None = None,
    k: float = RRF_K,
) -> dict[int, float]:
    """Fuse rank lists. Ranks, not scores, are comparable across retrievers."""
    weights = weights or (1.0,) * len(lists)
    fused: dict[int, float] = {}
    for ranked, weight in zip(lists, weights, strict=True):
        for rank_position, (chunk_id, _) in enumerate(ranked, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (k + rank_position)
    return fused


def _positions(ranked: Sequence[tuple[int, float]]) -> dict[int, int]:
    return {chunk_id: position for position, (chunk_id, _) in enumerate(ranked, start=1)}


def search(
    store: Store,
    query: str,
    *,
    provider: embedder.Embedder,
    mode: str = "hybrid",
    k: int = 8,
    paths: Sequence[str] = (),
    langs: Sequence[str] = (),
    require_all: bool = False,
    max_per_file: int = MAX_PER_FILE_DEFAULT,
    min_similarity: float = 0.0,
) -> list[Hit]:
    """Return the best chunks for a natural-language or identifier query."""
    pool = max(k * CANDIDATE_MULTIPLIER, 24)
    terms = lexicon.query_terms(query)

    lexical: list[tuple[int, float]] = []
    if mode in {"hybrid", "keyword"}:
        lexical = bm25.rank(store, terms, limit=pool, require_all=require_all)

    semantic: list[tuple[int, float]] = []
    if mode in {"hybrid", "semantic"}:
        provider.load()
        if provider.usable:
            ids, sims = vector_scores(store, provider.embed_one(query), paths=paths, langs=langs)
            semantic = top_by_score(ids, sims, pool)
            if min_similarity > 0:
                semantic = [(cid, s) for cid, s in semantic if s >= min_similarity]
        elif mode == "semantic":
            raise NoCandidates(provider.error or "embedding model unavailable")

    lists = [ranked for ranked in (lexical, semantic) if ranked]
    if not lists:
        return []
    fused = reciprocal_rank_fusion(lists)
    # One rank-1 hit out of `n` retrievers is the ceiling, so scores stay comparable
    # across modes: 100 means "first place everywhere".
    ceiling = len(lists) / (RRF_K + 1.0)
    semantic_at, lexical_at = _positions(semantic), _positions(lexical)

    picked: list[Hit] = []
    per_file: dict[str, int] = {}
    for chunk_id, score in sorted(fused.items(), key=lambda item: (-item[1], item[0])):
        row = store.chunk_row(chunk_id)
        if row is None:
            continue
        if paths and not path_matches(row["path"], paths):
            continue
        if langs and row["lang"] not in langs:
            continue
        if max_per_file:
            used = per_file.get(row["path"], 0)
            if used >= max_per_file:
                continue
            per_file[row["path"]] = used + 1
        picked.append(
            Hit(
                chunk_id=chunk_id,
                path=row["path"],
                start=row["start_line"],
                end=row["end_line"],
                symbol=row["symbol"],
                kind=row["kind"],
                text=row["text"],
                score=round(100.0 * score / ceiling, 2),
                semantic=semantic_at.get(chunk_id),
                lexical=lexical_at.get(chunk_id),
                lang=row["lang"],
                root=store.meta.get("root", ""),
                ranks=(semantic_at.get(chunk_id), lexical_at.get(chunk_id)),
            )
        )
        if len(picked) >= k:
            break
    return picked
