"""Okapi BM25 over the inverted index stored beside the vectors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .db import Store

K1 = 1.2
B = 0.75
# A term present in a quarter of all chunks carries almost no signal, and walking its
# postings is the expensive part -- so it only ranks when the query offers nothing else.
COMMON_DF_RATIO = 0.25
# In a folder of a dozen chunks every identifier clears 25%, so the ratio only starts
# meaning something once the corpus is big enough for it to exclude real signal.
MIN_DOCS_FOR_DF_RATIO = 24
MAX_POSTINGS = 500_000


@dataclass
class CorpusStats:
    docs: int
    avg_dl: float


def stats(store: Store) -> CorpusStats:
    """Chunk count and mean length, cached in meta at index time."""
    docs = int(store.meta.get("chunks", 0) or 0)
    avg_dl = float(store.meta.get("avg_dl", 0) or 0.0)
    if not docs or not avg_dl:
        row = store.conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(AVG(dl), 0) AS mean FROM chunk"
        ).fetchone()
        docs, avg_dl = row["n"], float(row["mean"]) or 1.0
    return CorpusStats(docs=docs, avg_dl=avg_dl or 1.0)


def _idf(df: int, docs: int) -> float:
    return math.log(1.0 + (docs - df + 0.5) / (df + 0.5))


def rank(
    store: Store,
    terms: Iterable[str],
    *,
    limit: int = 200,
    require_all: bool = False,
    allow: set[int] | None = None,
) -> list[tuple[int, float]]:
    """Score chunks against the query terms, best first.

    `allow` narrows the candidate set before scoring, which is how a --path filter stays
    cheap instead of rescanning the whole index.
    """
    wanted = list(dict.fromkeys(term for term in terms if term))
    if not wanted:
        return []
    info = stats(store)
    if info.docs == 0:
        return []

    scores: dict[int, float] = {}
    hits: dict[int, int] = {}
    usable_terms = 0

    for term in wanted:
        row = store.conn.execute(
            "SELECT df FROM term_stat WHERE term = ?", (term,)
        ).fetchone()
        if row is None or row["df"] <= 0:
            continue
        df = row["df"]
        if (
            len(wanted) > 1
            and info.docs >= MIN_DOCS_FOR_DF_RATIO
            and df / info.docs > COMMON_DF_RATIO
        ):
            continue

        idf = _idf(df, info.docs)
        usable_terms += 1
        postings = store.conn.execute(
            "SELECT p.chunk_id AS id, p.tf AS tf, c.dl AS dl"
            " FROM posting p JOIN chunk c ON c.id = p.chunk_id"
            " WHERE p.term = ? LIMIT ?",
            (term, MAX_POSTINGS),
        ).fetchall()
        for posting in postings:
            chunk_id = posting["id"]
            if allow is not None and chunk_id not in allow:
                continue
            length_penalty = K1 * (1.0 - B + B * (posting["dl"] or 1) / info.avg_dl)
            contribution = idf * posting["tf"] * (K1 + 1.0) / (posting["tf"] + length_penalty)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution
            hits[chunk_id] = hits.get(chunk_id, 0) + 1

    if require_all and usable_terms == len(wanted):
        scores = {cid: s for cid, s in scores.items() if hits[cid] >= len(wanted)}

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
