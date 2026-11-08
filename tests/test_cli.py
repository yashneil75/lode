"""End to end through the command line, with the model swapped for a deterministic fake."""

from __future__ import annotations

import json
import re

import pytest
import typer

from conftest import FakeEmbedder
from lode import cli, db, paths


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch):
    monkeypatch.setattr(cli.embedder, "Embedder", lambda *a, **k: FakeEmbedder())


def run(*args):
    try:
        cli.main([str(a) for a in args])
    except SystemExit as exc:  # click exits after every standalone invocation
        return int(exc.code or 0)
    return 0


def _bytes(match) -> float:
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    return float(match.group(1)) * units[match.group(2)]


@pytest.fixture
def indexed(sample_project, capsys):
    capsys.readouterr()
    assert run("index", str(sample_project), "-n", "demo") == 0
    capsys.readouterr()
    return sample_project


def test_index_then_search_round_trip(indexed, capsys):
    code = run("search", "hash_password", "-i", "demo")
    assert code == 0
    out = capsys.readouterr().out
    assert "src/auth.py" in out


def test_search_json_is_machine_readable(indexed, capsys):
    run("search", "token", "-i", "demo", "--json")
    payload = json.loads(capsys.readouterr().out)
    assert payload["index"] == "demo"
    assert isinstance(payload["hits"], list)
    assert {"path", "start", "end", "score"} <= set(payload["hits"][0])


def test_search_files_mode_prints_only_paths(indexed, capsys):
    run("search", "password", "-i", "demo", "-f")
    out = capsys.readouterr().out.strip().splitlines()
    assert out and all(line.strip().endswith((".py", ".js", ".md")) for line in out)


def test_search_reports_a_clear_miss(indexed, capsys):
    """Exit 1 on no match, like grep, so scripts can branch on it."""
    code = run("search", "zzzqqqxx", "-i", "demo", "--mode", "keyword")
    assert code == 1
    assert "no matches" in capsys.readouterr().out


def test_similarity_only_results_are_flagged(indexed, capsys):
    """A model that found nothing to match on should not pretend it agreed with itself."""
    run("search", "zzzqqqxx", "-i", "demo", "--mode", "semantic")
    out = capsys.readouterr().out
    assert "similarity alone" in out


def test_search_with_no_indexes_at_all_is_an_error(capsys):
    code = run("search", "anything", "-i", "missing")
    assert code != 0
    assert "index" in capsys.readouterr().err


def test_search_of_an_unknown_index_lists_what_exists(indexed, capsys):
    code = run("search", "anything", "-i", "nope")
    assert code != 0
    err = capsys.readouterr().err
    assert "nope" in err and "demo" in err


def test_list_shows_the_index(indexed, capsys):
    assert run("list") == 0
    out = capsys.readouterr().out
    assert "demo" in out and "test-model" in out and "files" in out


def test_index_reports_the_size_that_list_will_show(sample_project, capsys):
    """The write-ahead log is folded into the .lode file when the store closes.

    Measuring it before that happens prints a stub next to a big sidecar, so the two
    commands would disagree about how much disk an index costs.
    """
    capsys.readouterr()
    assert run("index", str(sample_project), "-n", "demo") == 0
    out = capsys.readouterr().out
    reported = re.search(r"index file ([\d.]+)(B|KB|MB|GB)", out)
    assert reported, out
    actual = paths.index_path("demo").stat().st_size
    assert _bytes(reported) == pytest.approx(actual, abs=1024)


def test_a_wide_terminal_gets_the_root_path(monkeypatch):
    """The path column is the first thing to go when the terminal is narrow."""
    import io

    from rich.console import Console

    from lode import output

    row = {
        "name": "demo", "files": 4, "chunks": 12, "model": "BAAI/bge-small-en-v1.5",
        "size_on_disk": 40_000, "updated_at": 0, "root": r"C:\code\repo\src",
    }
    for width, expected in ((70, False), (140, True)):
        buffer = io.StringIO()
        monkeypatch.setattr(output, "console", Console(file=buffer, width=width, highlight=False))
        output.render_indexes([row])
        rendered = buffer.getvalue()
        assert ("root" in rendered) is expected, rendered


def test_list_without_indexes_tells_you_what_to_do(capsys):
    assert run("list") == 0
    assert "lode index" in capsys.readouterr().out


def test_list_json(indexed, capsys):
    run("list", "--json")
    rows = json.loads(capsys.readouterr().out)
    assert [row["name"] for row in rows] == ["demo"]


def test_stats_reports_the_corpus(indexed, capsys):
    assert run("stats", "demo") == 0
    out = capsys.readouterr().out
    assert "demo" in out and "chunks" in out


def test_rename_keeps_the_index_searchable(indexed, capsys):
    assert run("rename", "demo", "demo2") == 0
    capsys.readouterr()
    assert run("search", "hash_password", "-i", "demo2") == 0
    assert not db.exists("demo")


def test_rename_refuses_to_clobber(indexed, capsys):
    run("index", str(indexed), "-n", "other")
    capsys.readouterr()
    assert run("rename", "other", "demo") != 0
    assert "already exists" in capsys.readouterr().out
    assert db.exists("other") and db.exists("demo")


def test_rename_reports_a_missing_source(indexed, capsys):
    assert run("rename", "ghost", "demo") != 0


def test_delete_removes_the_files(indexed):
    assert run("delete", "demo", "--yes") == 0
    assert not db.exists("demo")


def test_delete_all(indexed, sample_project, capsys):
    run("index", str(sample_project / "src"), "-n", "second")
    capsys.readouterr()
    assert run("delete", "--all", "--yes") == 0
    assert db.all_names() == []


def test_delete_unknown_name_is_a_non_zero_exit(indexed, capsys):
    assert run("delete", "nope", "--yes") != 0


def test_reindexing_the_same_folder_is_incremental(indexed, capsys):
    capsys.readouterr()
    assert run("index", str(indexed), "-n", "demo") == 0
    assert "up to date" in capsys.readouterr().out


def test_pointing_a_name_at_a_second_folder_needs_rebuild(indexed, sample_project, tmp_path, capsys):
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    capsys.readouterr()
    assert run("index", str(other), "-n", "demo") != 0
    assert "rebuild" in capsys.readouterr().out


def test_bad_index_name_is_rejected(indexed, capsys):
    assert run("index", str(indexed), "-n", "../escape") != 0


def test_quiet_index_prints_only_the_name(sample_project, capsys):
    assert run("index", str(sample_project), "-n", "shh", "-q") == 0
    assert capsys.readouterr().out.strip() == "shh"


def test_no_embed_still_searches(sample_project, capsys):
    assert run("index", str(sample_project), "-n", "lexical", "--no-embed") == 0
    capsys.readouterr()
    assert run("search", "hash_password", "-i", "lexical") == 0
    assert "src/auth.py" in capsys.readouterr().out


def test_unfetchable_weights_are_reported_not_crashed(sample_project, capsys, monkeypatch):
    """The first run on a machine with no route to a mirror must still produce an index.

    The handler formatted the exception as if it were a string, so the fallback path
    died with AttributeError instead of saying what went wrong.
    """

    class NoWeights(FakeEmbedder):
        def prepare(self, quiet: bool = False, force: bool = False):
            raise cli.modelsource.ModelUnavailable(
                "could not fetch tokenizer.json for BAAI/bge-small-en-v1.5\n"
                "https://hf-mirror.com/x: <urlopen error timed out>\n"
                "https://huggingface.co/x: RemoteDisconnected"
            )

    monkeypatch.setattr(cli.embedder, "Embedder", lambda *a, **k: NoWeights())
    assert run("index", str(sample_project), "-n", "demo") == 0
    out = capsys.readouterr().out
    assert "could not fetch weights" in out
    assert "could not fetch tokenizer.json for BAAI/bge-small-en-v1.5" in out
    assert "keyword-only" in out
    assert "AttributeError" not in out


def test_model_download_shares_which_mirrors_failed(capsys, monkeypatch):
    """The one command meant for debugging a fetch must not hide the reasons."""

    class NoWeights(FakeEmbedder):
        def prepare(self, quiet: bool = False, force: bool = False):
            raise cli.modelsource.ModelUnavailable(
                "could not fetch tokenizer.json for BAAI/bge-small-en-v1.5\n"
                "https://hf-mirror.com/x: <urlopen error timed out>\n"
                "https://huggingface.co/x: RemoteDisconnected"
            )

    monkeypatch.setattr(cli.embedder, "Embedder", lambda *a, **k: NoWeights())
    assert run("model", "download") == 4
    out = capsys.readouterr().out
    assert "hf-mirror" in out and "RemoteDisconnected" in out
    assert "LODE_MIRRORS" in out


def test_doctor_explains_the_setup(indexed, capsys):
    assert run("doctor") == 0
    assert "demo" in capsys.readouterr().out


def test_version(indexed, capsys):
    assert run("version") == 0
    assert capsys.readouterr().out.strip()


def test_bare_invocation_prints_help(capsys):
    assert run() == 0
    assert "lode" in capsys.readouterr().out


def test_dash_dash_version_is_shorthand(capsys):
    assert run("-v") == 0
    assert capsys.readouterr().out.strip().startswith("lode")


def test_unknown_command_exits_non_zero(capsys):
    assert run("nonsense") != 0


def test_path_filter_reaches_the_cli(indexed, capsys):
    run("search", "password", "-i", "demo", "-p", "login.js")
    out = capsys.readouterr().out
    assert "login.js" in out and "auth.py" not in out


def test_top_limits_results(indexed, capsys):
    run("search", "password", "-i", "demo", "-k", "1", "--json")
    assert len(json.loads(capsys.readouterr().out)["hits"]) == 1


def test_mode_keyword_is_accepted(indexed, capsys):
    assert run("search", "hash", "-i", "demo", "--mode", "keyword") == 0


def test_bad_mode_is_rejected(indexed, capsys):
    assert run("search", "hash", "-i", "demo", "--mode", "vibes") != 0
