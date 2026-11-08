"""Filesystem layout for lode.

Everything lode writes lives under one home directory, overridable with LODE_HOME.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ENV_HOME = "LODE_HOME"
_ENV_CACHE = "LODE_CACHE"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _xdg(key: str, fallback: Path) -> Path:
    value = os.environ.get(key)
    if value:
        return Path(value).expanduser()
    return fallback


def home() -> Path:
    """Root for indexes, model cache and settings."""
    override = os.environ.get(_ENV_HOME)
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        base = _xdg("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        return base / "lode"
    return _xdg("XDG_DATA_HOME", Path.home() / ".local" / "share") / "lode"


def indexes_dir() -> Path:
    return home() / "indexes"


def model_cache_dir() -> Path:
    override = os.environ.get(_ENV_CACHE)
    if override:
        return Path(override).expanduser().resolve()
    return home() / "models"


def index_path(name: str) -> Path:
    return indexes_dir() / f"{name}.lode"


def ensure_dirs() -> None:
    indexes_dir().mkdir(parents=True, exist_ok=True)
    model_cache_dir().mkdir(parents=True, exist_ok=True)


def version() -> str:
    from . import __version__

    return __version__
