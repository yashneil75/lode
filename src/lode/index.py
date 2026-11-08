"""Walking a repository and turning files into chunks, vectors and postings."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import pathspec

from . import chunk as chunking
from . import embedder, lexicon
from .db import Store

MAX_BYTES = 512_000
# Embeddings are gathered per wave so a whole monorepo never has to sit in RAM twice.
WAVE_BYTES = 4_000_000

DEFAULT_IGNORE = (
    ".git/", ".hg/", ".svn/", ".jj/",
    "node_modules/", "bower_components/", "vendor/", ".venv/", "venv/", "env/",
    "__pycache__/", "*.py[cod]", ".mypy_cache/", ".pytest_cache/", ".ruff_cache/",
    "dist/", "build/", "out/", "target/", ".next/", ".nuxt/", ".svelte-kit/",
    ".cache/", ".parcel-cache/", "coverage/", ".nyc_output/", "htmlcov/",
    ".idea/", ".vscode/", ".fleet/", "*.swp", "*.swo", ".DS_Store",
    "*.min.js", "*.min.css", "*.bundle.js", "*.map", "*.d.ts", "*.tsbuildinfo",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "Cargo.lock", "poetry.lock",
    "uv.lock", "composer.lock", "Gemfile.lock", "go.sum", "requirements.txt.lock",
    "shrinkwrap.yaml", "*.lock",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.ico", "*.svg", "*.pdf",
    "*.zip", "*.gz", "*.bz2", "*.xz", "*.7z", "*.tar", "*.rar",
    "*.woff", "*.woff2", "*.ttf", "*.otf", "*.eot", "*.mp3", "*.mp4", "*.mov", "*.avi",
    "*.so", "*.dylib", "*.dll", "*.exe", "*.class", "*.jar", "*.war", "*.o", "*.a",
    "*.obj", "*.pdb", "*.ilk", "*.pyd", "*.bin", "*.wasm", "*.sqlite", "*.db",
)

NULL_BYTE = b"\x00"
DECODE_ERRORS = "replace"


@dataclass
class Report:
    root: str
    name: str
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped_binary: int = 0
    skipped_large: int = 0
    chunks: int = 0
    embedded: int = 0
    seconds: float = 0.0
    model: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def touched(self) -> int:
        return self.added + self.updated

    @property
    def chunks_per_second(self) -> float:
        return self.chunks / self.seconds if self.seconds > 0.01 else 0.0

    @property
    def files(self) -> int:
        return self.added + self.updated + self.unchanged


class Ignores:
    """Default rules plus the repository's own .gitignore files, nested ones included."""

    def __init__(self, root: Path, extra: Sequence[str] = ()) -> None:
        self.root = root
        self.default = pathspec.GitIgnoreSpec.from_lines(
            [*DEFAULT_IGNORE, *extra, ".lode/"]
        )
        self._cache: dict[Path, pathspec.GitIgnoreSpec | None] = {}

    def _spec_for(self, directory: Path) -> pathspec.GitIgnoreSpec | None:
        if directory in self._cache:
            return self._cache[directory]
        spec: pathspec.GitIgnoreSpec | None = None
        ignore_file = directory / ".gitignore"
        if ignore_file.is_file():
            try:
                lines = ignore_file.read_text(encoding="utf-8", errors=DECODE_ERRORS).splitlines()
            except OSError:
                lines = []
            spec = pathspec.GitIgnoreSpec.from_lines(lines)
        self._cache[directory] = spec
        return spec

    def _ignored(self, rel: str, directory: Path, is_dir: bool) -> bool:
        target = rel + "/" if is_dir else rel
        if self.default.match_file(target):
            return True
        walked = directory
        while walked >= self.root:
            try:
                here = walked.relative_to(self.root).as_posix()
            except ValueError:
                break
            spec = self._spec_for(walked)
            if spec is not None:
                tail = rel[len(here):] if here and here != "." else rel
                if tail and spec.match_file(tail + ("/" if is_dir else "")):
                    return True
            if walked == self.root:
                break
            walked = walked.parent
        return False

    def ignore_dir(self, path: Path) -> bool:
        rel = path.relative_to(self.root).as_posix()
        return self._ignored(rel, path, True)

    def ignore_file(self, path: Path) -> bool:
        rel = path.relative_to(self.root).as_posix()
        return self._ignored(rel, path.parent, False)


def walk(root: Path, ignores: Ignores, *, follow_symlinks: bool = False) -> Iterator[Path]:
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink() and not follow_symlinks:
                continue
            if entry.is_dir(follow_symlinks=follow_symlinks):
                if path.name == ".git" or ignores.ignore_dir(path):
                    continue
                stack.append(path)
            elif entry.is_file(follow_symlinks=follow_symlinks):
                if not ignores.ignore_file(path):
                    yield path


def digest_of(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1 << 20)
                if not block:
                    break
                size += len(block)
                hasher.update(block)
    except OSError:
        return "", 0
    return hasher.hexdigest(), size


def read_source(path: Path, max_bytes: int) -> str | None:
    """File content as text, or None if it is binary or unreadable."""
    try:
        with path.open("rb") as handle:
            head = handle.read(8000)
            if NULL_BYTE in head:
                return None
            handle.seek(0)
            raw = handle.read(max_bytes + 1)
    except OSError:
        return None
    if len(raw) > max_bytes:
        return None
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    return raw.decode("utf-8", errors=DECODE_ERRORS)


def relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def to_blob(vector: np.ndarray) -> bytes:
    return np.ascontiguousarray(vector, dtype="<f4").tobytes()


Progress = Callable[[str, int, int], None]


def build(
    store: Store,
    root: Path,
    *,
    provider: embedder.Embedder | None = None,
    rebuild: bool = False,
    extensions: Sequence[str] = (),
    extra_ignore: Sequence[str] = (),
    max_bytes: int = MAX_BYTES,
    follow_symlinks: bool = False,
    use_embeddings: bool = True,
    progress: Progress | None = None,
) -> Report:
    """Create or refresh an index, re-embedding only files whose bytes changed."""
    started = time.time()
    root = root.resolve()
    report = Report(root=str(root), name=store.name)
    ignores = Ignores(root, extra_ignore)
    provider = provider or embedder.Embedder()
    report.model = provider.alias

    if rebuild:
        for row in store.conn.execute("SELECT path FROM file").fetchall():
            store.remove_file(row["path"])
        store.commit()

    # A pending work item is (relpath, digest, size, mtime, units, is_new).
    pending: list[tuple[str, str, int, float, list, bool]] = []
    pending_bytes = 0
    seen: set[str] = set()

    def flush() -> None:
        nonlocal pending, pending_bytes
        if not pending:
            return
        texts: list[str] = []
        for relpath, _digest, _size, _mtime, units, _new in pending:
            texts.extend(chunking.embedding_text(relpath, unit) for unit in units)

        vectors: list[bytes | None] = [None] * len(texts)
        if use_embeddings and texts:
            _load(provider, report)
            if provider.usable:
                for index, vector in enumerate(provider.embed(texts)):
                    vectors[index] = to_blob(vector)
                report.embedded += len(texts)

        offset = 0
        for relpath, digest, size, mtime, units, is_new in pending:
            take = len(units)
            lang = chunking.detect(relpath)
            lexicons = [
                lexicon.lexicon(relpath, unit.symbol, unit.kind, unit.text) for unit in units
            ]
            store.write_file(
                relpath,
                digest,
                size,
                mtime,
                lang,
                [(u.ord, u.start, u.end, u.symbol, u.kind, u.text) for u in units],
                vectors[offset : offset + take],
                lexicons,
            )
            offset += take
            report.chunks += take
            if is_new:
                report.added += 1
            else:
                report.updated += 1
        store.commit()
        pending, pending_bytes = [], 0

    for path in walk(root, ignores):
        if extensions:
            name = path.name.lower()
            suffix = path.suffix.lower()
            if suffix not in extensions and not any(name.endswith(e) for e in extensions):
                continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_size > max_bytes:
            report.skipped_large += 1
            continue

        relpath = relative(path, root)
        existing = store.file_row(relpath)
        if existing and not rebuild and _stamp_matches(existing, stat):
            seen.add(relpath)
            report.unchanged += 1
            continue

        digest, _ = digest_of(path)
        if not digest:
            continue
        if existing and existing["sha256"] == digest:
            # Content is identical and only the clock moved: keep chunks, refresh stamp.
            store.conn.execute("UPDATE file SET mtime = ? WHERE id = ?", (stat.st_mtime, existing["id"]))
            seen.add(relpath)
            report.unchanged += 1
            continue

        content = read_source(path, max_bytes)
        if content is None:
            report.skipped_binary += 1
            continue
        units = chunking.chunk(relpath, content)
        if not units:
            continue

        seen.add(relpath)
        pending.append((relpath, digest, stat.st_size, stat.st_mtime, units, existing is None))
        pending_bytes += stat.st_size
        if pending_bytes >= WAVE_BYTES:
            flush()
            if progress:
                progress("index", report.touched, report.files)

    flush()
    report.removed = len(store.prune_files(seen))
    _record(store, provider, report, started)
    return report


def _stamp_matches(row, stat: os.stat_result) -> bool:
    return abs(float(row["mtime"]) - stat.st_mtime) < 1e-3 and int(row["bytes"]) == stat.st_size


def _load(provider: embedder.Embedder, report: Report) -> None:
    """A missing model must not lose the lexical half of the index."""
    try:
        provider.load()
    except Exception as exc:  # pragma: no cover - depends on onnxruntime failures
        provider.error = f"{type(exc).__name__}: {exc}"
    if not provider.usable and provider.error:
        note = provider.error
        report.warnings.append(f"no embeddings: {_first_line(note)} (keyword ranking still works)")
        report.model = ""


def _first_line(text: str) -> str:
    return text.splitlines()[0] if text.splitlines() else text


def _record(store: Store, provider: embedder.Embedder, report: Report, started: float) -> None:
    counts = store.counts()
    store.set_meta(
        root=report.root,
        model=report.model,
        dim=str(provider.dim or ""),
        indexed_at=int(started),
        updated_at=int(time.time()),
        chunks=int(counts["chunks"]),
        avg_dl=f"{counts['avg_dl']:.4f}",
        files=int(counts["files"]),
        bytes=int(counts["bytes"]),
        terms=int(counts["terms"]),
        embedded=int(counts["embedded"]),
    )
    store.commit()
    report.seconds = time.time() - started
