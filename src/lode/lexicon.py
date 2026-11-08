"""Lexical analysis for code: the vocabulary BM25 ranks over.

Code identifiers are the whole game here. `HttpAuthMiddleware` must match a query for
"auth middleware", and `parse_url` must match "url parse", so every identifier is stored
both whole and split on camel/kebab/snake boundaries.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

# Words that appear in prose filenames and comments but carry no ranking signal.
STOP = frozenset(
    """
    a an and are as at be been but by do does for from had has have he her his how i if
    in into is it its no not of on or our so that the their them then there these they
    this to was we were what when which who will with would yes you your
    """.split()
)

# Whole identifiers: allow the punctuation that legitimately sits inside one.
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_+#.\-]*|\d[\d_]*")

# camelCase / PascalCase boundary, incl. HTTPServer -> HTTP, Server.
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z][a-z]*|[a-z]+|\d+")
_SPLIT = re.compile(r"[_\-.\s/]+")

MAX_TERM = 40
# A term in a path or symbol name says more about a chunk than a term in a comment does.
PATH_BOOST = 2
SYMBOL_BOOST = 2


def parts(word: str) -> list[str]:
    """`HttpAuth_middleware-v2` -> ['http', 'auth', 'middleware', 'v2']."""
    out: list[str] = []
    for piece in _SPLIT.split(word):
        if not piece:
            continue
        lowered = piece.lower()
        if len(piece) > 1 and not piece.islower():
            out.extend(m.lower() for m in _CAMEL.findall(piece) if len(m) > 1)
        out.append(lowered)
    seen: list[str] = []
    for token in out:
        if token not in seen:
            seen.append(token)
    return seen


def clean(token: str) -> str | None:
    token = token.strip("._-+#")
    if len(token) < 2 or len(token) > MAX_TERM:
        return None
    if token.isdigit():
        return None
    if token in STOP:
        return None
    return token.lower()


def terms(text: str) -> list[str]:
    """Every rankable term in order, duplicates kept."""
    found: list[str] = []
    for match in WORD.findall(text):
        for piece in parts(match):
            token = clean(piece)
            if token:
                found.append(token)
    return found


def counts(text: str) -> Counter:
    return Counter(terms(text))


def merge(*fields: Counter, weights: Iterable[int]) -> Counter:
    """Combine per-field term counts, weighting path and symbol fields above body."""
    total: Counter = Counter()
    for field, weight in zip(fields, weights, strict=True):
        total.update({term: tf * weight for term, tf in field.items()})
    return total


def lexicon(relpath: str, symbol: str, kind: str, body: str) -> Counter:
    """The rankable vocabulary of one chunk, with field boosts folded in."""
    return merge(
        counts(relpath),
        counts(f"{symbol} {kind}"),
        counts(body),
        weights=(PATH_BOOST, SYMBOL_BOOST, 1),
    )


def query_terms(text: str) -> list[str]:
    """Unique query terms, order preserved for stable RRF ties."""
    out: list[str] = []
    for token in terms(text):
        if token not in out:
            out.append(token)
    return out
