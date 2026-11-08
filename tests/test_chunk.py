"""Chunking must put every line somewhere and cut where a human would cut."""

from __future__ import annotations

from lode import chunk as chunking

BODY = "\n".join(f"    total = total + {i}" for i in range(chunking.MIN_LINES))

PY = f'''"""Module docstring."""


def alpha(value):
{BODY}
    return value + 1


class Beta:
{BODY}

    def gamma(self):
{BODY}
        return 3

    def delta(self):
{BODY}
        return 4
'''

JS = f"""export function epsilon(a, b) {{
{BODY}
  return a - b;
}}

function zeta(n) {{
{BODY}
  return n * 2;
}}

const eta = function (n) {{
{BODY}
  return n * 3;
}};
"""

MD = (
    "# Title\n\n" + "intro prose\n" * chunking.MIN_LINES + "\n"
    "## Setup\n\n" + "setup prose\n" * chunking.MIN_LINES + "\n"
    "## Usage\n\n" + "usage prose\n" * chunking.MIN_LINES
)


def covered_lines(units) -> set[int]:
    return {line for unit in units for line in range(unit.start, unit.end + 1)}


def test_python_splits_on_definitions():
    units = chunking.chunk("pkg/mod.py", PY)
    symbols = [unit.symbol for unit in units]
    assert "alpha" in symbols
    assert "Beta" in symbols
    assert "gamma" in symbols, "a class method deserves its own chunk"
    assert covered_lines(units) == set(range(1, len(PY.splitlines()) + 1))


def test_javascript_splits_on_functions():
    units = chunking.chunk("src/epsilon.js", JS)
    symbols = [unit.symbol for unit in units]
    assert "epsilon" in symbols
    assert "zeta" in symbols
    assert covered_lines(units) == set(range(1, len(JS.splitlines()) + 1))


def test_a_leading_docstring_does_not_steal_the_label():
    units = chunking.chunk("pkg/mod.py", '"""Notes."""\ndef alpha(value):\n    return value\n')
    assert len(units) == 1
    assert units[0].symbol == "alpha", "the chunk is about alpha, not the module docstring"


def test_a_leading_comment_does_not_steal_the_label():
    units = chunking.chunk("pkg/mod.py", "# explains the helper\ndef beta(value):\n    return v\n")
    assert units[0].symbol == "beta"


def test_a_commented_out_definition_never_names_a_chunk():
    units = chunking.chunk("x.py", "# def not_real():\ndef real():\n    return 1\n")
    assert units[0].symbol == "real"


def test_commented_out_definition_is_not_a_boundary():
    code = f"def real():\n{BODY}\n    return 1\n\n\n# def not_real():\n#     return 2\n"
    units = chunking.chunk("x.py", code)
    assert [unit.symbol for unit in units] == ["real"]


def test_markdown_splits_on_headings():
    units = chunking.chunk("README.md", MD)
    assert len(units) >= 2
    assert covered_lines(units) == set(range(1, len(MD.splitlines()) + 1))


def test_large_block_is_split_and_stays_within_the_limit():
    body = "\n".join(f"    value_{i} = {i}" for i in range(chunking.MAX_LINES * 2))
    units = chunking.chunk("x.py", f"def big():\n{body}\n")
    assert len(units) > 1
    for unit in units:
        assert unit.end - unit.start + 1 <= chunking.MAX_LINES + 4


def test_split_blocks_overlap_so_context_survives_the_cut():
    body = "\n".join(f"    value_{i} = {i}" for i in range(chunking.MAX_LINES * 3))
    units = chunking.chunk("x.py", f"def big():\n{body}\n")
    starts = [unit.start for unit in units]
    ends = [unit.end for unit in units]
    assert all(b <= a + 1 for a, b in zip(ends, starts[1:]))


def test_tiny_file_is_one_chunk():
    units = chunking.chunk("tiny.py", "x = 1\n")
    assert len(units) == 1
    assert units[0].start == 1 and units[0].end == 1


def test_empty_file_yields_nothing():
    assert chunking.chunk("empty.py", "") == []


def test_embed_text_prefixes_path_and_symbol():
    unit = chunking.Unit(ord=0, start=1, end=2, symbol="alpha", kind="def",
                         text="def alpha():\n    ...")
    text = chunking.embedding_text("pkg/mod.py", unit)
    assert text.startswith("pkg/mod.py alpha\n")


def test_units_are_numbered_dense_and_in_order():
    units = chunking.chunk("pkg/mod.py", PY)
    assert [unit.ord for unit in units] == list(range(len(units)))


def test_language_detection():
    assert chunking.detect("a/b/c.py") == "python"
    assert chunking.detect("a/b/c.tsx") == "tsx"
    assert chunking.detect("Makefile") == "make"
    assert chunking.detect("notes.txt") == "text"
