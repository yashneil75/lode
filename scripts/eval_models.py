"""Small labelled eval: which embedding model actually finds the right file?

Ground truth is the lode source itself, so the questions have one correct module each.
Run: python scripts/eval_models.py [--corpus src/lode]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("LODE_HOME", os.path.join(os.getcwd(), ".lodehome"))

from lode import db, embedder, index, search  # noqa: E402

CASES = [
    ("which mirror do we download the model weights from", "modelsource.py"),
    ("retry a partial download with backoff", "modelsource.py"),
    ("where is the data directory chosen on windows", "paths.py"),
    ("respect XDG_DATA_HOME if it is set", "paths.py"),
    ("split a file into chunks that follow function boundaries", "chunk.py"),
    ("track brace depth so we do not cut inside a function body", "chunk.py"),
    ("the inverted index postings and document frequency table", "db.py"),
    ("safely delete an index and its wal sidecars", "db.py"),
    ("okapi bm25 scoring with length normalisation", "bm25.py"),
    ("reciprocal rank fusion of two result lists", "search.py"),
    ("batch the cosine scan so memory stays bounded", "search.py"),
    ("camelCase identifier tokenisation for code", "lexicon.py"),
    ("skip a file that looks like a compiled binary", "index.py"),
    ("do not re-embed a file whose contents did not change", "index.py"),
    ("load the onnx session lazily and normalise vectors", "embedder.py"),
]

MODELS = [
    "BAAI/bge-small-en-v1.5",
    "snowflake/snowflake-arctic-embed-xs",
    "Qdrant/constella-zero",
    "",  # keyword only
]


def run(corpus: Path, alias: str, mode: str) -> tuple[float, float, float]:
    name = "eval" + (alias.split("/")[-1].replace(".", "-") or "keyword")
    provider = embedder.Embedder(alias or None)
    with db.Store(name) as store:
        started = time.time()
        index.build(store, corpus, provider=provider)
        build_seconds = time.time() - started
        hits = sum(1 for query, expected in CASES
                   if _top(store, query, expected, provider, mode))
        top3 = sum(1 for query, expected in CASES
                   if _within(store, query, expected, provider, mode, 3))
    accuracy = hits / len(CASES)
    return accuracy, top3 / len(CASES), build_seconds


def _top(store, query, expected, provider, mode) -> bool:
    try:
        hits = search.search(store, query, provider=provider, mode=mode, k=1)
    except search.NoCandidates:
        return False
    return bool(hits) and Path(hits[0].path).name == expected


def _within(store, query, expected, provider, mode, k: int) -> bool:
    try:
        hits = search.search(store, query, provider=provider, mode=mode, k=k)
    except search.NoCandidates:
        return False
    return any(Path(hit.path).name == expected for hit in hits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="src/lode")
    args = parser.parse_args()
    corpus = Path(args.corpus)

    print(f"corpus={corpus} cases={len(CASES)}\n")
    print(f"{'model':38} {'mode':9} {'top1':>6} {'top3':>6} {'index s':>8}")
    for alias in MODELS:
        modes = ["semantic", "hybrid"] if alias else ["keyword"]
        for mode in modes:
            top1, top3, seconds = run(corpus, alias, mode)
            label = alias or "(none: bm25 only)"
            print(f"{label:38} {mode:9} {top1:>6.0%} {top3:>6.0%} {seconds:>8.1f}")


if __name__ == "__main__":
    main()
