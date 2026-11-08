"""Terminal rendering: ranked hits, index tables, and machine-readable output."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from . import lexicon
from .search import Hit

console = Console(highlight=False, soft_wrap=False)
_THEME = "ansi_dark"

_LEXER_ALIAS = {
    "typescript": "typescript",
    "tsx": "tsx",
    "javascript": "javascript",
    "python": "python",
    "go": "go",
    "rust": "rust",
    "java": "java",
    "kotlin": "kotlin",
    "c": "c",
    "cpp": "cpp",
    "csharp": "c#",
    "ruby": "ruby",
    "php": "php",
    "swift": "swift",
    "scala": "scala",
    "dart": "dart",
    "sql": "sql",
    "shell": "bash",
    "yaml": "yaml",
    "toml": "toml",
    "json": "json",
    "markdown": "markdown",
    "html": "html",
    "css": "css",
    "vue": "html",
    "svelte": "html",
    "graphql": "graphql",
    "lua": "lua",
    "elixir": "elixir",
    "haskell": "haskell",
    "protobuf": "protobuf",
}
_DEFAULT_SNIPPET_LINES = 6
_BAR = "█"
_BAR_EMPTY = "░"
_LEXER_CACHE: dict[str, str] = {}


def _lexer_for(lang: str) -> str:
    """A pygments lexer name that exists, or plain text."""
    if lang in _LEXER_CACHE:
        return _LEXER_CACHE[lang]
    candidate = _LEXER_ALIAS.get(lang, "text")
    try:
        get_lexer_by_name(candidate)
    except ClassNotFound:
        candidate = "typescript" if candidate == "tsx" else "text"
        try:
            get_lexer_by_name(candidate)
        except ClassNotFound:
            candidate = "text"
    _LEXER_CACHE[lang] = candidate
    return candidate


def _bar(score: float, width: int = 10) -> Text:
    filled = max(0, min(width, round(width * score / 100.0)))
    return Text(_BAR * filled + _BAR_EMPTY * (width - filled), style="bold cyan")


def _stamp(seconds: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(seconds))


def _ago(seconds: float) -> str:
    """Compact 'updated' cell -- absolute stamps push the path column off the line."""
    if not seconds:
        return ""
    age = max(0.0, time.time() - seconds)
    if age < 90:
        return "just now"
    if age < 5_400:
        return f"{int(age // 60)}m ago"
    if age < 129_600:
        return f"{int(age // 3_600)}h ago"
    if age < 518_400:
        return f"{int(age // 86_400)}d ago"
    return _stamp(seconds)[:10]


def _short_path(text: str, width: int) -> str:
    """Home-relative, then tail-kept -- the interesting end of a path is its last part."""
    if not text:
        return ""
    home = str(Path.home())
    if text == home:
        text = "~"
    elif text.startswith(home):
        text = "~" + text[len(home) :]
    return _shorten(text, width)


def _matched_terms(query: str, text: str, limit: int = 6) -> list[str]:
    """Which query terms actually occur in this chunk -- the 'why' behind a lexical hit."""
    present = lexicon.terms(text)
    wanted = [term for term in lexicon.query_terms(query) if term in present]
    return wanted[:limit]


def render_hits(
    hits: Sequence[Hit],
    *,
    query: str,
    index_name: str,
    mode: str,
    snippet: bool = True,
    snippet_lines: int = _DEFAULT_SNIPPET_LINES,
    show_ranks: bool = True,
    elapsed: float = 0.0,
) -> None:
    if not hits:
        console.print(f"[dim]no matches in [bold]{index_name}[/bold] for {query!r}[/dim]")
        return
    for position, hit in enumerate(hits, start=1):
        location = f"{hit.path}:{hit.start}-{hit.end}" if hit.lines > 1 else f"{hit.path}:{hit.start}"
        header = Text()
        header.append(f"{position:>2}  ", style="bold white")
        header.append(location, style="bold blue")
        if hit.symbol:
            header.append(f"  {hit.symbol}", style="green")
        if hit.kind:
            header.append(f" [{hit.kind}]", style="dim")
        header.append("  ")
        header.append_text(_bar(hit.score))
        header.append(f" {hit.score:5.1f}", style="cyan")
        console.print(header)

        if not snippet:
            continue
        body = hit.text.split("\n")
        shown = body[: max(1, snippet_lines)]
        truncated = len(body) - len(shown)
        snippet = Syntax(
            "\n".join(shown),
            _lexer_for(hit.lang),
            theme=_THEME,
            line_numbers=True,
            start_line=hit.start,
            indent_guides=False,
            background_color="default",
        )
        console.print(Padding(snippet, (0, 0, 0, 4)))
        if truncated > 0:
            console.print(
                Text(f"      +{truncated} more line" + ("s" if truncated != 1 else ""), style="dim italic")
            )

        notes: list[str] = []
        matched = _matched_terms(query, hit.text)
        if matched:
            notes.append("matched " + ", ".join(matched))
        if show_ranks and hit.semantic:
            notes.append(f"semantic #{hit.semantic}")
        if show_ranks and hit.lexical:
            notes.append(f"lexical #{hit.lexical}")
        if notes:
            console.print("      " + " \u00b7 ".join(notes), style="dim")

    summary = f"{len(hits)} hits in [bold]{index_name}[/bold] via [bold]{mode}[/bold]"
    if elapsed:
        summary += f" \u00b7 {elapsed * 1000:.0f} ms"
    console.print()
    console.print(summary, style="dim")


def render_json(hits: Sequence[Hit], *, query: str, index_name: str, mode: str, elapsed: float) -> None:
    payload = {
        "query": query,
        "index": index_name,
        "mode": mode,
        "seconds": round(elapsed, 4),
        "hits": [
            {
                **{key: value for key, value in asdict(hit).items() if key != "ranks"},
                "lines": hit.lines,
            }
            for hit in hits
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def render_paths(hits: Sequence[Hit], *, unique: bool) -> None:
    seen: set[str] = set()
    for hit in hits:
        if unique and hit.path in seen:
            continue
        seen.add(hit.path)
        print(hit.path)


def render_indexes(rows: Sequence[dict]) -> None:
    if not rows:
        console.print(
            Panel(
                "No indexes yet.\n\n  [bold]lode index[/] [dim]# index the folder you are in[/]\n"
                "  [bold]lode index ~/code/my-app -n my-app[/]",
                title="lode",
                border_style="dim",
            )
        )
        return
    cells = [
        (
            str(row["name"]),
            f"{row['files']:,}",
            f"{row['chunks']:,}",
            _short_model(row.get("model", "")) or "keyword-only",
            _human(row.get("size_on_disk", 0)),
            _ago(float(row.get("updated_at") or 0)),
            _short_path(row.get("root", ""), 38),
        )
        for row in rows
    ]
    headers = ("name", "files", "chunks", "model", "size", "updated", "root")
    widths = [max(len(h), *(len(c[i]) for c in cells)) for i, h in enumerate(headers)]
    # Rich only leaves a column its leftover width when the others are pinned, so the
    # path absorbs whatever the terminal has room for instead of being squeezed.
    pinned = sum(widths[:-1]) + 2 * (len(headers) - 1)
    room = console.width - pinned - 1
    show_root = room >= 16
    table = Table(show_header=True, header_style="bold dim", box=None, pad_edge=False)
    for index, header in enumerate(headers):
        if index == len(headers) - 1 and not show_root:
            break
        width = widths[index] if index < len(headers) - 1 else room
        table.add_column(
            header,
            justify="right" if index in (1, 2, 4) else "left",
            width=width,
            no_wrap=True,
            overflow="ellipsis",
        )
    for cell in cells:
        table.add_row(
            Text(cell[0], style="bold"),
            cell[1],
            cell[2],
            cell[3],
            cell[4],
            cell[5],
            *([Text(cell[6], style="dim")] if show_root else []),
        )
    console.print(table)


def render_stats(row: dict) -> None:
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column("field", style="dim")
    table.add_column("value", justify="left")
    first = str(row.get("indexed_at") or "")
    last = str(row.get("updated_at") or "")
    for key, value in row.items():
        if key == "updated_at" and first == last:
            continue  # a freshly built index has nothing to show twice
        table.add_row(str(key), _format_value(key, value))
    console.print(table)


def _format_value(key: str, value: object) -> str:
    if key in {"bytes", "size_on_disk"}:
        return _human(int(value or 0))
    if key.endswith("_at"):
        return _epoch(int(value or 0))
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _epoch(seconds: int) -> str:
    if not seconds:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(seconds))


def _short_model(alias: str) -> str:
    return alias.split("/")[-1] if alias else ""


def _shorten(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    head = Path(text).name
    keep = max(1, width - len(head) - 4)
    return "…" + text[-keep:] if keep < len(text) else text


def _human(nbytes: int) -> str:
    value = float(nbytes or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"
