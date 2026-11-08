"""Fetching the embedding model weights.

fastembed's own downloader goes through the Hugging Face Hub API, which fails in a
surprising number of real environments (corporate egress filters, regional blocks,
proxied CI). lode therefore fetches the plain files it needs over a configurable
mirror chain and hands the finished directory to fastembed, so an index can be built
wherever the weights can be reached -- and reused wherever they cannot.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from . import paths

MANIFEST = "lode-model.json"
_MIRROR_KEY = "LODE_MIRRORS"
_TIMEOUT = float(os.environ.get("LODE_DOWNLOAD_TIMEOUT", "60"))
_RETRIES = int(os.environ.get("LODE_DOWNLOAD_RETRIES", "3"))
_CHUNK = 1 << 16

Progress = Callable[[str, int, int], None]


class ModelUnavailable(RuntimeError):
    """No mirror could supply the weights; search falls back to keyword mode."""


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    hf_repo: str
    revision: str
    weight_file: str
    required: tuple[str, ...]
    optional: tuple[str, ...]
    dim: int
    max_tokens: int
    note: str
    size_gb: float = 0.0

    @property
    def dir_name(self) -> str:
        return f"{self.hf_repo.replace('/', '--')}-{self.revision}"


# Files every ONNX sentence model needs alongside its weights.
_SIDE_CAR_REQUIRED = ("config.json", "tokenizer.json")
_SIDE_CAR_OPTIONAL = (
    "tokenizer_config.json",
    "special_tokens_map.json",
    "modules.json",
)


def _descriptions():
    from fastembed.text.onnx_embedding import supported_onnx_models

    return supported_onnx_models


def _registry() -> dict[str, ModelSpec]:
    """Build specs from fastembed's own table so versions stay in sync."""
    specs: dict[str, ModelSpec] = {}
    for entry in _descriptions():
        sources = getattr(entry, "sources", None)
        repo = getattr(sources, "hf", None) if sources else None
        if not repo:
            continue
        extra = tuple(getattr(entry, "additional_files", ()) or ())
        specs[entry.model] = ModelSpec(
            alias=entry.model,
            hf_repo=repo,
            revision="main",
            weight_file=entry.model_file,
            required=(entry.model_file, *_SIDE_CAR_REQUIRED, *extra),
            optional=_SIDE_CAR_OPTIONAL,
            dim=entry.dim,
            max_tokens=getattr(entry, "max_length", 512) or 512,
            note=(entry.description or "").strip(),
            size_gb=float(getattr(entry, "size_in_GB", 0.0) or 0.0),
        )
    return specs


def available() -> dict[str, ModelSpec]:
    try:
        return _registry()
    except Exception:  # fastembed not importable in this build
        return {}


# The three that matter for code: fast/defaults and one step up in accuracy.
SHORTLIST = (
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
)


def canonical(alias: str) -> str:
    """Accept short names like 'bge-small' so the CLI stays pleasant to type."""
    specs = available()
    if alias in specs:
        return alias
    needle = alias.lower().replace("_", "-")
    matches = [
        name
        for name in specs
        if needle in name.lower() or name.lower().endswith("/" + needle)
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        # 'bge-small' should mean the current default, not the older v0 model that
        # also matches the prefix.
        for preferred in SHORTLIST:
            if preferred in matches:
                return preferred
        return sorted(matches)[0]
    return alias


def shortlist() -> list[ModelSpec]:
    specs = available()
    return [specs[name] for name in SHORTLIST if name in specs]


def resolve_spec(alias: str) -> ModelSpec:
    specs = available()
    name = canonical(alias)
    if name in specs:
        return specs[name]
    raise ModelUnavailable(
        f"unknown model {alias!r}. try: " + ", ".join(sorted(s for s in specs)[:6])
    )


def model_dir(spec: ModelSpec) -> Path:
    return paths.model_cache_dir() / spec.dir_name


def default_mirrors() -> list[str]:
    raw = os.environ.get(_MIRROR_KEY)
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    # hf-mirror.com serves the same tree as huggingface.co and stays reachable where
    # the primary host is reset at the TCP layer.
    return [
        "https://hf-mirror.com/{repo}/resolve/{rev}/{path}",
        "https://huggingface.co/{repo}/resolve/{rev}/{path}",
    ]


def _urls(spec: ModelSpec, file: str, mirrors: list[str]) -> Iterator[str]:
    for mirror in mirrors:
        template = mirror if "{" in mirror else mirror.rstrip("/") + "/{path}"
        yield template.format(repo=spec.hf_repo, rev=spec.revision, path=file)


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, _RETRIES + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": f"lode/{paths.version()}"}
            )
            with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                with tmp.open("wb") as handle:
                    while True:
                        block = response.read(_CHUNK)
                        if not block:
                            break
                        handle.write(block)
                        done += len(block)
                        if total:
                            _notify(url, done, total)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt < _RETRIES:
                time.sleep(0.5 * 2 ** (attempt - 1))
    raise ModelUnavailable(f"{url}: {last_error}")


_NOTIFIER: Progress | None = None


def _notify(url: str, done: int, total: int) -> None:
    if _NOTIFIER is not None:
        _NOTIFIER(url, done, total)


def set_progress(callback: Progress | None) -> None:
    global _NOTIFIER
    _NOTIFIER = callback


def is_ready(spec: ModelSpec) -> bool:
    directory = model_dir(spec)
    return (directory / MANIFEST).is_file() and all(
        (directory / name).is_file() for name in spec.required
    )


def ensure(
    alias: str,
    *,
    mirrors: list[str] | None = None,
    force: bool = False,
    quiet: bool = False,
) -> Path:
    """Return a local directory holding the model, downloading what is missing."""
    spec = resolve_spec(alias)
    directory = model_dir(spec)
    if is_ready(spec) and not force:
        return directory

    if not quiet:
        set_progress(_report)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        chain = mirrors or default_mirrors()
        failures: list[str] = []
        for name in spec.required:
            target = directory / name
            if target.is_file() and not force:
                continue
            for url in _urls(spec, name, chain):
                try:
                    _fetch(url, target)
                    break
                except ModelUnavailable as exc:
                    failures.append(str(exc))
            else:
                raise ModelUnavailable(
                    f"could not fetch {name} for {alias}\n" + "\n".join(failures[-6:])
                )
        for name in spec.optional:
            target = directory / name
            if target.is_file():
                continue
            for url in _urls(spec, name, chain):
                try:
                    _fetch(url, target)
                    break
                except ModelUnavailable:
                    continue
        (directory / MANIFEST).write_text(
            json.dumps(
                {
                    "alias": spec.alias,
                    "repo": spec.hf_repo,
                    "revision": spec.revision,
                    "dim": spec.dim,
                    "weight_file": spec.weight_file,
                },
                indent=2,
            )
        )
    finally:
        if not quiet:
            set_progress(None)
    return directory


def import_from(local: Path, alias_hint: str | None = None) -> Path:
    """Copy a prepared ONNX directory into lode's cache, skipping the network."""
    weights = list(local.glob("*.onnx"))
    if not weights:
        raise ModelUnavailable(f"no .onnx weights found in {local}")
    spec_name = alias_hint or local.name
    spec = resolve_spec(spec_name) if alias_hint else None
    dest = model_dir(spec) if spec else paths.model_cache_dir() / local.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(local, dest)
    (dest / MANIFEST).write_text(
        json.dumps({"alias": spec_name, "imported_from": str(local)}, indent=2)
    )
    return dest


def _report(url: str, done: int, total: int) -> None:
    import sys

    name = url.rsplit("/", 1)[-1]
    pct = done * 100 // total
    sys.stderr.write(
        f"\r  {name:<28} {done / 1e6:6.1f}/{total / 1e6:5.1f} MB {pct:3d}%"
    )
    if done >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()
