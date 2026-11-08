"""Throughput probe: how chunk width and truncation change embedding speed.

Not shipped with the binary; used to pick defaults and to keep the README honest.
Run: python scripts/bench_embed.py
"""

from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("LODE_HOME", os.path.join(os.getcwd(), ".lodehome"))

from fastembed import TextEmbedding  # noqa: E402
from lode import chunk as chunking  # noqa: E402
from lode import paths  # noqa: E402


def corpus(widths=(24, 48, 96)) -> dict[int, list[str]]:
    buckets: dict[int, list[str]] = {w: [] for w in widths}
    files = glob.glob(os.path.join(".venv", "Lib", "site-packages", "**", "*.py"), recursive=True)
    for path in files[:400]:
        try:
            lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
        except OSError:
            continue
        for width in widths:
            for start in range(0, max(1, len(lines) - width), width):
                buckets[width].append("\n".join(lines[start : start + width]))
    return buckets


def main() -> None:
    model_dir = next(iter(paths.model_cache_dir().glob("*bge-small*")))
    model = TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        specific_model_path=str(model_dir),
        lazy_load=False,
        cache_dir=str(paths.model_cache_dir()),
        threads=8,
    )
    buckets = corpus()
    print(f"{'width':>6} {'trunc':>7} {'avg chars':>10} {'chunks/s':>9}")
    for width, texts in sorted(buckets.items()):
        for truncation in (None, 2400, 1200, 600):
            sample = [(t if truncation is None else t[:truncation]) for t in texts][:400]
            if not sample:
                continue
            average = sum(len(t) for t in sample) / len(sample)
            started = time.time()
            list(model.embed(sample, batch_size=64))
            rate = len(sample) / (time.time() - started)
            print(f"{width:>6} {str(truncation or 'off'):>7} {average:>10.0f} {rate:>9.1f}")


if __name__ == "__main__":
    main()
