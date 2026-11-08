"""Split source files into chunks that stay inside one definition where possible.

Line-window chunking smears a function across two documents and confuses both BM25 and
the embedding model, so lode finds definition boundaries first and only falls back to a
window for prose, data files, and definitions too large to fit in one chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_LINES = 48
MAX_LINES = 140
MIN_LINES = 8
OVERLAP_LINES = 14
MAX_CHUNKS_PER_FILE = 4000

LANG_BY_EXT: dict[str, str] = {
    ".py": "python", ".pyi": "python", ".ipynb": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".mts": "typescript", ".cts": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".hh": "cpp", ".cs": "csharp", ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".scala": "scala", ".sc": "scala", ".dart": "dart", ".lua": "lua", ".pl": "perl",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".fish": "shell",
    ".sql": "sql", ".vue": "vue", ".svelte": "svelte", ".astro": "astro",
    ".html": "html", ".htm": "html", ".css": "css", ".scss": "scss", ".less": "less",
    ".toml": "toml", ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".jsonc": "json",
    ".md": "markdown", ".mdx": "markdown", ".rst": "rst", ".txt": "text",
    ".proto": "protobuf", ".graphql": "graphql", ".tf": "terraform",
    ".gradle": "groovy", ".groovy": "groovy", ".make": "make", ".mk": "make",
    ".dockerfile": "dockerfile", ".ex": "elixir", ".exs": "elixir", ".erl": "erlang",
    ".hs": "haskell", ".ml": "ocaml", ".nim": "nim", ".zig": "zig", ".jl": "julia",
    ".r": "r", ".m": "matlab", ".f90": "fortran", ".vim": "vim",
}

BRACE_LANGS = frozenset(
    {"javascript", "typescript", "tsx", "go", "rust", "java", "kotlin", "c", "cpp",
     "csharp", "php", "swift", "scala", "dart", "vue", "svelte", "astro", "css", "scss",
     "less", "json", "groovy", "elixir", "terraform", "c"}
)
INDENT_LANGS = frozenset({"python", "ruby", "shell", "yaml", "haskell", "r", "julia"})

# `def name`, `class Name`, `func (s *T) Name`, `fn main`, `public void Name` ...
DEF_RE = re.compile(
    r"""^\s*(?:@\w[\w.]*
        | (?:(?:public|private|protected|internal|export|default|static|final|abstract|
             sealed|open|implicit|inline|async|virtual|override|const|readonly|pub|unsafe|
             extern|template|auto|mut|async)\s+)*
            (?:def|class|struct|enum|interface|trait|impl|fn|func|function|type|record|
             module|namespace|contract|macro|fun|test|describe)\b
      )
    """,
    re.VERBOSE,
)

# `const handler = async (req) => {`, `let Widget = function () {}`, `var Main = class {}`
ASSIGN_FN_RE = re.compile(
    r"^\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function\b|[A-Za-z_$][\w$]*\s*=>|\([^)]*\)\s*=>|class\b)"
)

# `def name`, `class Name`, `func (s *T) Name`, `fn main` -- keyword then identifier.
_NAME_RE = re.compile(
    r"\b(?:def|class|struct|enum|interface|trait|impl|fn|func|function|type|record|"
    r"module|namespace|contract|macro|fun)\s+(?:\([^)]*\)\s*)?([A-Za-z_$][\w$]*)"
)
# `void doThing()`, `getUsers(req)` -- a bare name heading a definition line.
_CALL_RE = re.compile(
    r"^\s*(?:@[\w.]+\s+)*(?:(?:public|private|protected|internal|static|final|abstract|"
    r"override|virtual|async|const|let|var|readonly|pub|export|default|inline|template|"
    r"auto|unsafe|extern|open|sealed)\s+)*(?:[\w<>\[\].*&]+\s+)?([A-Za-z_$][\w$]*)\s*(?:\(|=)"
)

# Control flow and keywords that head a line but name nothing.
_NOT_A_SYMBOL = frozenset(
    {
        "if", "elif", "else", "for", "while", "until", "switch", "match", "case", "catch",
        "except", "finally", "return", "yield", "do", "try", "with", "typeof", "await",
        "super", "then", "when", "unless", "loop", "repeat", "begin", "end", "elsif",
        "rescue", "ensure", "new", "delete", "print", "raise", "assert", "import",
        "from", "package", "package_name", "use", "require", "include", "module",
    }
)

# A jest/pytest block's name lives in its string argument, not after a keyword.
_TEST_TITLE_RE = re.compile(
    r"""^\s*(?:test|describe|context|spec|it)\s*\(\s*(['"`])(?P<title>.{1,90}?)\1"""
)

_KIND_TYPE = re.compile(
    r"\b(?:class|struct|interface|trait|enum|impl|record|type|module|namespace|contract)\b"
)
_KIND_TEST = re.compile(r"\b(?:test|describe|it|spec|context)\b[\s(\"']")

HEADING_RE = re.compile(r"^(?P<hashes>#{1,4})\s+(?P<title>.+?)\s*$")
SECTION_RE = re.compile(r"^\s*(?:##|=\s|\.\s{2,})")


@dataclass(frozen=True)
class Unit:
    ord: int
    start: int  # 1-based, inclusive
    end: int  # 1-based, inclusive
    symbol: str
    kind: str
    text: str


def detect(relpath: str, first_line: str = "") -> str:
    lowered = relpath.lower()
    ext = lowered[lowered.rfind(".") :] if "." in lowered.rsplit("/", 1)[-1] else ""
    lang = LANG_BY_EXT.get(ext)
    if lang:
        return lang
    base = lowered.rsplit("/", 1)[-1]
    if base.startswith("dockerfile"):
        return "dockerfile"
    if base.startswith("makefile"):
        return "make"
    if base.endswith(".lock") or base in {"cmakelists.txt", "build"}:
        return "text"
    if first_line.startswith("#!") or first_line.startswith("<?php"):
        return "shell"
    return "text"


def _depth_delta(line: str, in_block: bool) -> tuple[int, bool]:
    """Change in brace depth from one line, ignoring braces in strings and comments."""
    delta = 0
    quote = ""
    escaped = False
    index = 0
    while index < len(line):
        char = line[index]
        nxt = line[index + 1 : index + 2]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\":
            escaped = True
            index += 1
            continue
        if in_block:
            if nxt == "*/":
                in_block = False
                index += 2
                continue
            index += 1
            continue
        if quote:
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in "\"'`":
            quote = char
            index += 1
            continue
        if char == "/" and nxt == "*":
            in_block = True
            index += 2
            continue
        if char == "/" and nxt == "/":
            break
        if char == "#":
            break
        if char == "{":
            delta += 1
        elif char == "}":
            delta -= 1
        index += 1
    return delta, in_block


def _depths(lines: list[str], braces: bool) -> list[int]:
    """Depth *before* each line is emitted, so a line at depth 0 is top level."""
    out: list[int] = []
    depth = 0
    in_block = False
    for line in lines:
        out.append(depth)
        if braces:
            delta, in_block = _depth_delta(line, in_block)
            depth = max(0, depth + delta)
    return out


def _kind(line: str) -> str:
    if _KIND_TYPE.search(line):
        return "type"
    if _KIND_TEST.search(line):
        return "test"
    return "fn"


def _symbol(line: str) -> tuple[str, str]:
    """Best name to label this chunk with, plus the kind of thing it is."""
    match = _NAME_RE.search(line)
    if match and match.group(1).lower() not in _NOT_A_SYMBOL:
        return match.group(1), _kind(line)

    match = _TEST_TITLE_RE.match(line)
    if match:
        return match.group("title").strip(), "test"

    match = ASSIGN_FN_RE.match(line)
    if match:
        return match.group(1), "fn"

    match = _CALL_RE.match(line)
    if match and match.group(1).lower() not in _NOT_A_SYMBOL:
        name = match.group(1)
        if _KIND_TYPE.search(line):
            return name, "type"
        return name, _kind(line)

    match = DEF_RE.match(line)
    if match:
        return "", _kind(line)
    return "", ""


def _boundaries(lines: list[str], depths: list[int], lang: str) -> list[int]:
    """Line indices that start a new definition."""
    marks: list[int] = []
    if lang in {"markdown", "rst", "text"}:
        for index, line in enumerate(lines):
            if HEADING_RE.match(line) or (lang != "text" and SECTION_RE.match(line)):
                marks.append(index)
        return marks or [0]

    # Python and friends nest only by indentation, so a method sits at indent 4 and is
    # still its own chunk; brace languages must be checked against real depth.
    indent_based = lang in INDENT_LANGS
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if not indent_based and depths[index] != 0:
            continue
        if DEF_RE.match(line) or ASSIGN_FN_RE.match(line):
            marks.append(index)
    return marks or [0]


_COMMENT_START = ("#", "//", "/*", "*", "--", ";", "'''", '"""', "<!--", "=pod")


def _label_line(lines: list[str], start: int, end: int) -> int:
    """Index of the first line that is code.

    A docstring or a commented-out definition glued to the front of a chunk must not
    steal its label -- the symbol field is boosted in the lexicon, so a wrong one
    quietly mis-ranks the chunk.
    """
    for index in range(start, end):
        stripped = lines[index].strip()
        if stripped and not stripped.startswith(_COMMENT_START):
            return index
    return start


def _merged_blocks(marks: list[int], total: int) -> list[tuple[int, int]]:
    """Contiguous [start, end) line spans between boundaries, covering the whole file."""
    spans: list[tuple[int, int]] = []
    if marks[0] > 0:
        spans.append((0, marks[0]))
    for index, start in enumerate(marks):
        end = marks[index + 1] if index + 1 < len(marks) else total
        spans.append((start, end))
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and (merged[-1][1] - merged[-1][0]) < MIN_LINES and (end - start) < MIN_LINES:
            merged[-1] = (merged[-1][0], end)
        elif (end - start) == 0:
            continue
        else:
            merged.append((start, end))
    # A trailing run of tiny definitions reads better as one chunk than as twenty.
    if len(merged) > 1:
        tail = merged[-1]
        if tail[1] - tail[0] < MIN_LINES:
            merged[-2] = (merged[-2][0], tail[1])
            merged.pop()
    return merged or [(0, total)]


def _split_large(span: tuple[int, int], lines: list[str], depths: list[int]) -> list[tuple[int, int]]:
    start, end = span
    if end - start <= MAX_LINES:
        return [(start, end)]
    # One level deeper counts too: the methods of a large class are the boundary we want.
    limit = depths[start] + 1
    inner = [
        index
        for index in range(start + 1, end)
        if (DEF_RE.match(lines[index]) or ASSIGN_FN_RE.match(lines[index]))
        and depths[index] <= limit
    ]
    if inner:
        spans = []
        cursor = start
        for mark in inner:
            if mark - cursor >= MIN_LINES:
                spans.append((cursor, mark))
                cursor = mark
        spans.append((cursor, end))
        if all(e - s <= MAX_LINES for s, e in spans):
            return spans
    # No usable inner boundary: window it, carrying context across the seam.
    spans = []
    step = TARGET_LINES
    cursor = start
    while cursor < end:
        stop = min(cursor + TARGET_LINES, end)
        spans.append((cursor, stop))
        if stop == end:
            break
        cursor = max(cursor + 1, stop - OVERLAP_LINES)
    return spans


def _text(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start:end])


def chunk(relpath: str, content: str, lang: str | None = None) -> list[Unit]:
    """Every line of the file lands in at least one unit; boundaries follow definitions."""
    lang = lang or detect(relpath, content.splitlines()[0] if content else "")
    lines = content.splitlines()
    total = len(lines)
    if total == 0:
        return []

    depths = _depths(lines, lang in BRACE_LANGS)
    marks = _boundaries(lines, depths, lang)
    spans: list[tuple[int, int]] = []
    for span in _merged_blocks(marks, total):
        spans.extend(_split_large(span, lines, depths))

    units: list[Unit] = []
    for ord_, (start, end) in enumerate(spans[:MAX_CHUNKS_PER_FILE]):
        if end <= start:
            continue
        body = _text(lines, start, end)
        if not body.strip():
            continue
        symbol, kind = _symbol(lines[_label_line(lines, start, end)])
        units.append(Unit(ord=ord_, start=start + 1, end=end, symbol=symbol, kind=kind, text=body))

    # Renumber after empties were dropped so `ord` stays dense and gaps stay honest.
    return [
        Unit(ord=index, start=u.start, end=u.end, symbol=u.symbol, kind=u.kind, text=u.text)
        for index, u in enumerate(units)
    ]


def embedding_text(relpath: str, unit: Unit) -> str:
    """What the model sees: path and symbol context wrapped around the code itself."""
    header = relpath if not unit.symbol else f"{relpath} {unit.symbol}"
    return f"{header}\n{unit.text}"
