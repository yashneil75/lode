"""Reading and writing lode's small JSON settings file."""

from __future__ import annotations

import json
from typing import Any

from . import paths

KEYS = ("default_model", "default_index")


def file():
    return paths.home() / "config.json"


def load() -> dict[str, Any]:
    try:
        raw = file().read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def save(**items: Any) -> None:
    current = load()
    current.update(items)
    file().parent.mkdir(parents=True, exist_ok=True)
    file().write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")


def get(key: str, default: Any = None) -> Any:
    return load().get(key, default)
