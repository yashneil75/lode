"""The `lode` command line."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import config, db, embedder, index as indexer
from . import modelsource, output, paths, search
from .db import Store

app = typer.Typer(
    name="lode",
    help="Semantic + keyword search over your codebases, in one small binary.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
model = typer.Typer(help="Choose, fetch and inspect the embedding model.", no_args_is_help=True)
app.add_typer(model, name="model")

console = Console(highlight=False)
_IS_TTY = sys.stdout.isatty()


# --------------------------------------------------------------------------- helpers


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-.")
    return (cleaned or "default")[:64]


def _select_index(explicit: str) -> tuple[str, str]:
    """Return (name, reason). Mirrors how git picks a repo: explicit, then cwd, then only child."""
    names = db.all_names()
    if not names:
        raise typer.BadParameter(
            "no indexes exist yet -- run `lode index` first", param_hint="--index"
        )
    if explicit:
        if not db.exists(explicit):
            raise typer.BadParameter(
                f"no index called {explicit!r}. available: " + ", ".join(names),
                param_hint="--index",
            )
        return explicit, "chosen"

    cwd = Path.cwd().resolve()
    anchored = []
    for name in names:
        with Store(name, readonly=True) as store:
            root = store.meta.get("root", "")
        if root:
            candidate = Path(root).resolve()
            if candidate == cwd or candidate in cwd.parents or cwd in candidate.parents:
                anchored.append(name)
    if len(anchored) == 1:
        return anchored[0], "matched your folder"
    if len(names) == 1:
        return names[0], "your only index"
    if anchored:
        raise typer.BadParameter(
            "several indexes cover this folder: " + ", ".join(anchored) + " -- pass --index",
            param_hint="--index",
        )
    raise typer.BadParameter(
        "more than one index exists, and none matches your folder -- pass --index ("
        + ", ".join(names) + ")",
        param_hint="--index",
    )


def _progress_factory(label: str):
    """A one-line status ticker, suppressed when output is piped."""
    if not _IS_TTY:
        return None

    def report(stage: str, done: int, total: int) -> None:
        console.file.write(f"\r  {label}: {done:,} files touched ({total:,} seen)")
        console.file.flush()

    return report


def _humanize(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 90:
        return f"{seconds:.1f} s"
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes}m {remainder:02d}s"


def _estimate(remaining_chunks: int, rate: float) -> str:
    if rate <= 0:
        return ""
    return _humanize(remaining_chunks / rate)


# --------------------------------------------------------------------------- commands


@app.command("index")
def index_command(
    path: Path = typer.Argument(Path.cwd(), help="Folder to index. Defaults to where you are."),
    name: str = typer.Option("", "--name", "-n", help="Index name. Defaults to the folder name."),
    rebuild: bool = typer.Option(False, "--rebuild", help="Re-chunk and re-embed every file."),
    model_alias: str = typer.Option("", "--model", "-m", help="Embedding model to use."),
    extensions: list[str] = typer.Option(
        [], "--ext", "-e", help="Only index these extensions, e.g. -e .py -e .ts (repeatable)."
    ),
    ignore: list[str] = typer.Option(
        [], "--ignore", "-x", help="Extra gitignore-style patterns to skip (repeatable)."
    ),
    max_bytes: int = typer.Option(512_000, "--max-bytes", help="Skip files larger than this."),
    no_embed: bool = typer.Option(
        False, "--no-embed", help="Keyword-only index: no model, no download, very fast."
    ),
    follow_symlinks: bool = typer.Option(False, "--follow-symlinks", help="Descend into symlinks."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print the index name."),
) -> None:
    """Create an index, or update an existing one in place.

    Re-running is cheap: files whose bytes are unchanged are skipped entirely.
    """
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise typer.BadParameter(f"{root} is not a folder", param_hint="PATH")

    # A name the user typed is theirs: refuse it rather than quietly rename their index.
    # Only a derived folder name gets slugged ("My App" -> "my-app").
    if name:
        chosen = name
        if _slug(chosen) != chosen:
            raise typer.BadParameter(
                f"invalid index name {name!r}: use letters, digits, dot, dash or underscore",
                param_hint="--name",
            )
    else:
        chosen = _slug(root.name)
    exts = tuple(e if e.startswith(".") else "." + e for e in extensions)
    provider = embedder.Embedder(model_alias or None)

    fresh = not db.exists(chosen)
    with Store(chosen) as store:
        previous_root = store.meta.get("root", "")
        if previous_root and Path(previous_root).resolve() != root and not rebuild:
            console.print(
                f"[yellow]index [bold]{chosen}[/bold] already points at {previous_root}.[/yellow]\n"
                "  same name, different folder: pass [bold]--rebuild[/bold] to retarget, or [bold]-n[/bold] a new name."
            )
            raise typer.Exit(2)

        if not no_embed and not quiet:
            console.print(f"  model [bold]{provider.alias.split('/')[-1]}[/bold]", style="dim")
            try:
                info = provider.prepare(quiet=not _IS_TTY)
                console.print(
                    f"  weights ready: {info.mb:g} MB, {info.dim} dims", style="dim"
                )
            except modelsource.ModelUnavailable as exc:
                console.print(f"  [yellow]could not fetch weights:[/] {_first_line(exc)}")
                console.print("  continuing with keyword-only ranking (BM25).")

        started = time.time()
        report = indexer.build(
            store,
            root,
            provider=provider,
            rebuild=rebuild,
            extensions=exts,
            extra_ignore=ignore,
            max_bytes=max_bytes,
            follow_symlinks=follow_symlinks,
            use_embeddings=not no_embed,
            progress=_progress_factory("scanning"),
        )
        if _IS_TTY:
            console.file.write("\r" + " " * 78 + "\r")
            console.file.flush()
        counts = store.counts()

    # SQLite folds the write-ahead log back into the main file when the last connection
    # closes, so measuring inside the `with` reports a stub and a big -wal.
    disk_bytes = store.path.stat().st_size

    if quiet:
        print(chosen)
        return

    verb = "indexed" if fresh else ("updated" if report.touched or report.removed else "up to date")
    console.print(
        f"\n  [bold green]{verb}[/bold green] [bold]{chosen}[/bold] → {root}"
    )
    console.print(
        "  "
        + f"{counts['files']:,} files \u00b7 {counts['chunks']:,} chunks \u00b7 "
        f"{counts['terms']:,} terms \u00b7 {output._human(counts['bytes'])} of source"
    )
    detail = []
    if report.added:
        detail.append(f"{report.added:,} new")
    if report.updated:
        detail.append(f"{report.updated:,} changed")
    if report.unchanged:
        detail.append(f"{report.unchanged:,} unchanged")
    if report.removed:
        detail.append(f"{report.removed:,} removed")
    if report.skipped_binary:
        detail.append(f"{report.skipped_binary:,} binary")
    if report.skipped_large:
        detail.append(f"{report.skipped_large:,} oversize")
    if not detail:
        detail = ["no indexable files found"]
    console.print("  " + " · ".join(detail))
    if report.chunks:
        console.print(
            f"  {_humanize(report.seconds)} · {report.chunks_per_second:.0f} chunks/s · "
            f"index file {output._human(disk_bytes)}"
        )
    for warning in report.warnings:
        console.print(f"  [yellow]{warning}[/yellow]")
    console.print(f"\n  try: [bold]lode search \"what you are looking for\" -i {chosen}[/bold]\n")


@app.command("search")
def search_command(
    query: str = typer.Argument(..., help="Plain words or an identifier fragment."),
    index_name: str = typer.Option("", "--index", "-i", help="Which index to search."),
    top: int = typer.Option(8, "--top", "-k", help="How many hits to show."),
    mode: str = typer.Option(
        "hybrid", "--mode", help="hybrid, semantic or keyword (hybrid fuses both with RRF)."
    ),
    paths_filter: list[str] = typer.Option(
        [], "--path", "-p", help="Restrict to paths matching a glob, e.g. -p 'src/**/auth*'."
    ),
    langs: list[str] = typer.Option([], "--lang", help="Restrict to languages, e.g. --lang python."),
    snippet_lines: int = typer.Option(6, "--snippet-lines", "-s", help="Lines of code per hit (0 hides code)."),
    all_terms: bool = typer.Option(False, "--all-terms", help="Require every query term to appear."),
    max_per_file: int = typer.Option(3, "--max-per-file", help="Cap hits from one file (0 disables)."),
    min_similarity: float = typer.Option(0.0, "--min-similarity", help="Drop weak semantic matches (0-1)."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
    files_only: bool = typer.Option(False, "--files", "-f", help="Just the matching file paths."),
) -> None:
    """Search inside an index."""
    if mode not in {"hybrid", "semantic", "keyword"}:
        raise typer.BadParameter("choose hybrid, semantic or keyword", param_hint="--mode")
    if top < 1:
        raise typer.BadParameter("--top must be at least 1", param_hint="--top")

    name, reason = _select_index(index_name)
    started = time.time()

    with Store(name, readonly=True) as store:
        stored_model = store.meta.get("model", "")
        counts = store.counts()
        if counts["chunks"] == 0:
            console.print(f"[yellow]index {name} is empty[/yellow] -- run `lode index {store.meta.get('root', '')}`")
            raise typer.Exit(1)

        if mode in {"hybrid", "semantic"} and not stored_model:
            if mode == "semantic":
                console.print(
                    f"[yellow]{name} was built keyword-only[/yellow] (no vectors stored)."
                    " Re-index without --no-embed for semantic search."
                )
                raise typer.Exit(3)
            mode = "keyword"

        provider = embedder.Embedder(stored_model or None)
        if mode != "keyword":
            provider.load(quiet=not _IS_TTY)
            if not provider.usable:
                if mode == "semantic":
                    console.print(f"[red]cannot load the model:[/] {_first_line(provider.error or '')}")
                    raise typer.Exit(3)
                console.print(
                    f"[dim]model unavailable ({_first_line(provider.error or 'unknown')}),"
                    " falling back to keyword ranking[/dim]"
                )
                mode = "keyword"

        try:
            hits = search.search(
                store,
                query,
                provider=provider,
                mode=mode,
                k=top,
                paths=paths_filter,
                langs=langs,
                require_all=all_terms,
                max_per_file=max_per_file if max_per_file >= 0 else 3,
                min_similarity=min_similarity,
            )
        except search.NoCandidates as exc:
            console.print(f"[red]{_first_line(str(exc))}[/red]")
            raise typer.Exit(3) from exc

        elapsed = time.time() - started
        if as_json:
            output.render_json(hits, query=query, index_name=name, mode=mode, elapsed=elapsed)
        elif files_only:
            output.render_paths(hits, unique=True)
        else:
            output.render_hits(
                hits,
                query=query,
                index_name=name,
                mode=mode,
                snippet=snippet_lines > 0,
                snippet_lines=snippet_lines,
                elapsed=elapsed,
            )
        if provider.error and not as_json and not files_only:
            console.print(f"  [dim]note: {reason} · {provider.error}[/dim]")
        if hits and not as_json and not files_only and all(h.lexical is None for h in hits):
            console.print("  [dim]no keywords matched -- ranked by similarity alone[/dim]")

    if not hits:
        raise typer.Exit(1)


@app.command("list")
def list_command(
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Every index you have: what it covers and how big it is."""
    import json as _json

    rows = []
    for name in db.all_names():
        try:
            with Store(name, readonly=True) as store:
                counts = store.counts()
                rows.append(
                    {
                        "name": name,
                        "root": store.meta.get("root", ""),
                        "model": store.meta.get("model", ""),
                        "updated_at": int(store.meta.get("updated_at", 0) or 0),
                        "size_on_disk": store.path.stat().st_size,
                        **counts,
                    }
                )
        except Exception as exc:  # a half-written index should not hide the rest
            rows.append({"name": name, "root": f"(unreadable: {exc})", "chunks": 0, "files": 0})
    if as_json:
        print(_json.dumps(rows, indent=2))
        return
    output.render_indexes(rows)


@app.command("rename")
def rename_command(
    old: str = typer.Argument(..., help="Existing index name."),
    new: str = typer.Argument(..., help="New name."),
) -> None:
    """Rename an index. Nothing is re-embedded."""
    if not db.exists(old):
        console.print(f"[red]no index called {old!r}[/red]")
        raise typer.Exit(2)
    target = _slug(new)
    if target != new:
        console.print(f"[dim]normalised the new name to {target!r}[/dim]")
    try:
        with Store(old) as store:
            store.rename(target)
    except FileExistsError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    console.print(f"  renamed [bold]{old}[/bold] \u2192 [bold green]{target}[/bold green]")


@app.command("delete")
def delete_command(
    names: list[str] = typer.Argument(None, help="One or more indexes to delete."),
    everything: bool = typer.Option(False, "--all", help="Delete every index."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask."),
) -> None:
    """Delete an index. The code it describes is untouched."""
    if everything:
        targets = db.all_names()
    else:
        targets = list(dict.fromkeys(names or []))
        if not targets:
            raise typer.BadParameter("give one or more index names, or --all", param_hint="NAMES")
        missing = [candidate for candidate in targets if not db.exists(candidate)]
        if missing:
            console.print(f"[red]no index called {', '.join(missing)}[/red]")
            raise typer.Exit(2)

    if not targets:
        console.print("  nothing to delete")
        return
    if not yes:
        summary = ", ".join(targets)
        typer.confirm(f"delete {len(targets)} index/es: {summary}?", abort=True)

    freed = 0
    for candidate in targets:
        with Store(candidate) as store:
            freed += store.path.stat().st_size if store.path.exists() else 0
            store.destroy()
    console.print(f"  deleted {len(targets)} index/es, reclaimed {output._human(freed)}")


@app.command("stats")
def stats_command(
    name: str = typer.Argument("", help="Index name (omit if you have only one)."),
) -> None:
    """Deep numbers on one index."""
    chosen, _ = _select_index(name)
    with Store(chosen, readonly=True) as store:
        counts = store.counts()
        meta = dict(store.meta)
        biggest = store.conn.execute(
            "SELECT path, chunks, bytes FROM file ORDER BY chunks DESC LIMIT 5"
        ).fetchall()
        langs = store.conn.execute(
            "SELECT f.lang AS lang, COUNT(*) AS n FROM chunk c"
            " JOIN file f ON f.id = c.file_id GROUP BY f.lang ORDER BY n DESC LIMIT 6"
        ).fetchall()
    info = {
        "name": chosen,
        **{k: v for k, v in meta.items() if k not in {"schema"}},
        **counts,
    }
    output.render_stats(info)
    if langs:
        console.print("\n  [bold]chunks by language[/bold]")
        for row in langs:
            console.print(f"    {row['lang']:14} {row['n']:>8,}")
    if biggest:
        console.print("\n  [bold]largest files[/bold]")
        for row in biggest:
            console.print(f"    {row['chunks']:>5,} chunks  {output._human(row['bytes']):>9}  {row['path']}")


@model.command("list")
def model_list(
    all_models: bool = typer.Option(False, "--all", help="Every model fastembed can run, not just the shortlist."),
) -> None:
    """Which embedding models lode can use, and what is already downloaded."""
    specs = modelsource.available()
    keep = {spec.alias for spec in modelsource.shortlist()}
    rows = []
    for alias, spec in sorted(specs.items(), key=lambda item: item[1].size_gb):
        if not all_models and alias not in keep:
            continue
        rows.append(
            {
                "alias": alias,
                "dim": spec.dim,
                "mb": round(spec.size_gb * 1000),
                "downloaded": modelsource.is_ready(spec),
                "default": alias == embedder.DEFAULT_MODEL,
            }
        )
    table = Table(box=None, header_style="bold dim", pad_edge=False)
    table.add_column("model", overflow="fold")
    table.add_column("dims", justify="right")
    table.add_column("MB", justify="right")
    table.add_column("on disk")
    for row in rows:
        table.add_row(
            Text(row["alias"], style="bold" if row["default"] else "")
            + Text("  (default)" if row["default"] else ""),
            str(row["dim"]),
            str(row["mb"]),
            Text("\u2713" if row["downloaded"] else "\u00b7", style="green" if row["downloaded"] else "dim"),
        )
    console.print(table)
    console.print(
        f"\n  cache: [dim]{paths.model_cache_dir()}[/dim]\n"
        "  set the default: [bold]lode model use bge-small[/bold]"
    )


@model.command("use")
def model_use(
    alias: str = typer.Argument(..., help="Model name or short form, e.g. bge-small."),
) -> None:
    """Pick the model used for new indexes."""
    try:
        spec = modelsource.resolve_spec(alias)
    except modelsource.ModelUnavailable as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    config.save(default_model=spec.alias)
    console.print(f"  new indexes will use [bold]{spec.alias}[/bold] ({spec.dim} dims)")


@model.command("download")
def model_download(
    alias: str = typer.Argument("", help="Model to fetch. Omit to fetch the default."),
    force: bool = typer.Option(False, "--force", help="Fetch again even if it is cached."),
) -> None:
    """Fetch the weights now, so the first index does not wait."""
    wanted = alias or config.get("default_model") or embedder.DEFAULT_MODEL
    try:
        info = embedder.Embedder(wanted).prepare(force=force)
    except modelsource.ModelUnavailable as exc:
        console.print(f"[red]{_first_line(str(exc))}[/red]")
        console.print(
            "  behind a network filter? set [bold]LODE_MIRRORS[/bold] to a reachable host, e.g.\n"
            "  LODE_MIRRORS=https://hf-mirror.com/{repo}/resolve/{rev}/{path}"
        )
        raise typer.Exit(4) from exc
    console.print(
        f"  [green]ready[/green] {info.alias} \u00b7 {info.mb:g} MB \u00b7 {info.dim} dims \u00b7 "
        f"cached at {paths.model_cache_dir()}"
    )


@app.command("doctor")
def doctor_command() -> None:
    """Report what lode sees: build, storage, models, indexes."""
    frozen = paths.is_frozen()
    table = Table(box=None, show_header=False, pad_edge=False, title=None)
    table.add_column("check", style="dim")
    table.add_column("state")
    table.add_row("version", f"{paths.version()} ({'binary' if frozen else 'python ' + sys.version.split()[0]})")
    table.add_row("home", str(paths.home()))
    indexes = db.all_names()
    table.add_row("indexes", ", ".join(indexes) if indexes else "none yet")
    provider = embedder.Embedder()
    try:
        spec = modelsource.resolve_spec(provider.alias)
        state = "ready" if modelsource.is_ready(spec) else "not downloaded"
    except modelsource.ModelUnavailable as exc:
        spec, state = None, _first_line(str(exc))
    table.add_row("default model", f"{provider.alias} \u00b7 {state}")
    with _memory_store() as mem:
        table.add_row("storage", f"sqlite {sqlite_version(mem)} \u00b7 numpy {numpy_version()}")
    table.add_row("mirrors", "\n".join(modelsource.default_mirrors()))
    console.print(table)
    if state == "ready":
        console.print(
            "\n  next: [bold]lode index[/bold] a folder, then [bold]lode search[/bold] it."
        )
    else:
        console.print(
            "\n  run [bold]lode model download[/bold] to fetch the weights,"
            " then [bold]lode index[/bold] a folder you want to search."
        )


def numpy_version() -> str:
    import numpy

    return numpy.__version__


def _memory_store():
    import contextlib
    import sqlite3

    @contextlib.contextmanager
    def opener():
        conn = sqlite3.connect(":memory:")
        try:
            yield conn
        finally:
            conn.close()

    return opener()


def sqlite_version(conn) -> str:
    return conn.execute("SELECT sqlite_version()").fetchone()[0]


@app.command("version")
def version_command() -> None:
    """Print the version."""
    print(f"lode {paths.version()}")


def _first_line(text: str) -> str:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    return lines[0][:200] if lines else "unknown error"


def _configure_streams() -> None:
    """Keep Unicode output from crashing a cp1252 console.

    Windows shells still hand us the active code page for piped output, and lode prints
    box-drawing and arrows; replacing an unencodable glyph beats dying mid-command.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def main(argv: Optional[list[str]] = None) -> None:
    _configure_streams()
    try:
        paths.ensure_dirs()
    except OSError as exc:  # a read-only home is survivable for `lode version`
        console.print(f"[yellow]cannot create {paths.home()}:[/] {exc}")
    app(_normalise(list(argv) if argv is not None else sys.argv[1:]))


def _normalise(args: list[str]) -> list[str]:
    """`lode` alone should behave like `lode --help`, and bare `--version` should work."""
    if not args:
        return ["--help"]
    if args[0] in {"-v", "--version"}:
        return ["version"]
    return args


if __name__ == "__main__":
    main()
