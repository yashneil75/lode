# lode

**Hybrid code search in one binary.** BM25 keyword ranking plus local embeddings,
fused into a single ranked answer. One index per codebase, one file on disk, no
server, no API key, nothing leaves your machine.

Named after the load you carry into a codebase you don't know yet.

```sh
lode index ~/code/my-app -n my-app
lode search "how do we retry a failed download" -i my-app
```

Real output, from lode's own source (this is the shape, not a staged demo):

```
 1  chunk.py:1-52  ██████████  99.2
      1 """Split source files into chunks that stay inside one definition where
      2
      3 Line-window chunking smears a function across two documents and confuse
      4 the embedding model, so lode finds definition boundaries first and only
      5 window for prose, data files, and definitions too large to fit in one c
      6 """
      +46 more lines
      matched chunks, split, definition, boundaries · semantic #1 · lexical #2

1 hits in lode-src via hybrid · 933 ms
```

The two retrievers agree or disagree in ways you can see: every hit says which terms
matched and where each list ranked it, so you can tell a confident answer from one the
model guessed at.

## Install

One command, no Python needed:

```sh
# macOS or Linux
curl -fsSL https://raw.githubusercontent.com/yashneil75/lode/main/install.sh | sh

# Windows (PowerShell)
irm https://raw.githubusercontent.com/yashneil75/lode/main/install.ps1 | iex
```

The installers pick your platform build, verify its SHA256 against `sha256sums.txt`
from the same release, and only then replace an existing install. Prebuilt archives are
on the [releases page](https://github.com/yashneil75/lode/releases):

| Build | Archive |
| --- | --- |
| Linux x86_64 | `lode-<version>-linux-x86_64.tar.gz` |
| Linux arm64 | `lode-<version>-linux-arm64.tar.gz` |
| macOS (Apple silicon) | `lode-<version>-darwin-arm64.tar.gz` |
| Windows x86_64 | `lode-<version>-windows-x86_64.zip` |

Or install from source with pipx, which needs Python 3.10+:

```sh
pipx install git+https://github.com/yashneil75/lode.git
```

Not on PyPI yet — the git URL is the working install path.

No Intel macOS build — that machine is old enough that `pipx` is the better answer.

## Commands

| | |
| --- | --- |
| `lode index [PATH]` | Create or update an index. `-n NAME`, `-e .py`, `-x 'docs/**'`, `--no-embed`, `--rebuild` |
| `lode search "QUERY"` | Search. `-i NAME`, `-k 8`, `--mode hybrid\|semantic\|keyword`, `-p 'src/**'`, `--lang python`, `--json`, `-f`, `--all-terms` |
| `lode list` | Every index: files, chunks, model, size, age, root |
| `lode rename OLD NEW` | Rename an index. No re-embedding |
| `lode delete NAME…` \| `--all` | Delete one or more indexes. Your code is untouched |
| `lode stats NAME` | Chunk counts per language, largest files, term count |
| `lode model list\|use\|download` | Choose the embedding model |
| `lode doctor` | What lode sees: build, home, indexes, model state, mirrors |

`search` exits `1` on no matches and `0` on hits, like `grep`, so it composes in
scripts:

```sh
lode search "csrf middleware" -i my-app -f | xargs -o nvim
```

## How it works

1. **Chunk.** Files split at definition boundaries (regex per language, then
   brace-depth or indent tracking so a cut never lands inside a function body), aiming
   for ~48 lines, with 14 lines of overlap so a question straddling a boundary still
   matches. The label is taken from the first *code* line, so a docstring or commented-out
   definition can't claim a chunk's symbol name.
2. **Two views of every chunk.** A bag of code terms (`camelCase`, `snake_case`,
   `kebab-case` all split; path and symbol fields boosted) goes into an inverted index;
   the text goes to the embedding model.
3. **Rank twice.** Okapi BM25 (`k1=1.2`, `b=0.75`) over the postings, cosine similarity
   over unit-normalised vectors.
4. **Fuse.** Reciprocal Rank Fusion, `1/(60+rank)` per list. Ranks, not raw scores,
   cross the boundary between retrievers, so a scale change in one cannot silently
   dominate the other. Displayed scores are normalised so `100.0` means *first place in
   every retriever that ran*.

Everything lives in one SQLite file per index (WAL, `posting` table without rowid,
vectors as BLOBs), so `lode list` is honest about cost and copying an index to a
coworker is a file copy.

Embedding is batched in 4 MB waves and the vector scan uses keyset pagination, so
memory stays flat on repos much bigger than RAM allows for the source alone.

## Numbers

Measured on a Windows laptop, CPU only, no GPU, lode 0.1.0.

**Accuracy.** `scripts/eval_models.py` asks 15 hand-written questions against lode's
own source (14 files, 168 chunks) and checks whether the expected module wins. This
corpus is small and written by the tool's author, so treat it as a smell test, not a
benchmark — the honest signal is the *ordering* between models, not the absolute
percentages.

| Model | Mode | top-1 | top-3 |
| --- | --- | --- | --- |
| `BAAI/bge-small-en-v1.5` (default, 67 MB) | semantic | **80%** | 93% |
| `BAAI/bge-small-en-v1.5` | hybrid | 73% | 93% |
| `snowflake/snowflake-arctic-embed-xs` | semantic | 53% | 73% |
| `snowflake/snowflake-arctic-embed-xs` | hybrid | 73% | 93% |
| `Qdrant/constella-zero` | semantic | 53% | 93% |
| `Qdrant/constella-zero` | hybrid | 67% | 93% |
| none (BM25 only) | keyword | 67% | 87% |

bge-small wins the default slot on semantic top-1 by a wide margin. On this 15-case
set, hybrid and semantic are within one question of each other — too few cases to call
a winner, which is exactly why `--mode` is exposed instead of hidden behind a tunable
fusion weight.

**Unseen code.** Indexed `pathspec` (29 files, 205 chunks, 127 KB) — a library lode has
never seen — and hand-checked six natural-language questions. All six top hits were the
right code: `negate a gitignore pattern with an exclamation mark` →
`patterns/gitignore/spec.py:pattern_to_regex`, `convert a glob into a regular
expression` → `patterns/gitignore/base.py:_translate_segment_glob`. A keyword query for
`PathResult` — a class *removed* in this version — ranked its replacement
(`util.py:TreeEntry.__init__`, fields `name`/`path`/`lstat`) first.

**Speed.**

| What | Cost |
| --- | --- |
| First index, 168 chunks (96 KB) | 24 s |
| First index, 205 chunks (127 KB) | 25 s |
| Re-index with nothing changed, 29 files | 0.9 s |
| Hybrid query, end to end | 1.0 s |
| Keyword-only query, end to end | 0.7 s |
| `lode list` | 0.5 s |

Roughly 8 chunks/s single-threaded on CPU: a mid-size repo is minutes, not hours. The
query numbers are whole-process costs from the standalone binary, and about 0.5 s of
every one is interpreter and library startup (`lode list` loads no model and still
takes that long); loading the ONNX session adds the rest. Batch related questions with
`--json` rather than paying that per question.

**Size.** An index file lands around 6.6× the source it describes (127 KB of pathspec →
840 KB index). The binaries are 153 MB unpacked / ~64 MB compressed because onnxruntime,
numpy and the Rust tokenizer ship inside. Model weights (67 MB) are *not* bundled —
they download to the lode home on first use, which is also how you pick a different
model.

## Configuration

| Variable | Effect |
| --- | --- |
| `LODE_HOME` | Where indexes, models and settings live. Default `~/.local/share/lode`, `%LOCALAPPDATA%\lode` on Windows |
| `LODE_MODEL` | Override the default embedding model for every command |
| `LODE_NO_EMBED` | Never load a model: keyword-only, fully offline |
| `LODE_CACHE` | Separate folder for model weights |
| `LODE_MIRRORS` | Comma-separated weight hosts. Default tries `hf-mirror.com` then `huggingface.co` |
| `LODE_DOWNLOAD_TIMEOUT` / `LODE_DOWNLOAD_RETRIES` | Fetch tuning |

A machine with no route to Hugging Face still works: `lode model download` once while
online, or index with `--no-embed` and lose only the semantic half of the ranking.

## Why not grep, or a vector DB?

`grep` is exact and unbeatable at identifiers you already know. It has no answer to
"where do we handle a failed download", and no notion of which of 40 hits is the one
you wanted. A hosted vector database solves recall but wants an API key, an embed
service, and your source uploaded somewhere. lode keeps the inverted index and the
vectors in the same SQLite file on your disk, and ranks with both.

## Build from source

```sh
git clone https://github.com/yashneil75/lode && cd lode
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
pytest                                    # hermetic: no network, no model needed
pyinstaller --noconfirm lode.spec         # -> dist/lode/lode
```

The test suite monkeypatches the embedder with a deterministic fake, so it runs hermetically
in any environment; `scripts/eval_models.py` is the only thing that needs real weights.

Releases build the same spec across four platforms in
[`.github/workflows/release.yml`](.github/workflows/release.yml) and publish archives
plus `sha256sums.txt` to a GitHub Release on every `v*` tag.

## Limits

- Single-file SQLite index: good to a few tens of thousands of files. Not a team server.
- Ranking is per-chunk text plus identifier structure. There is no call graph, so
  "what breaks if I change this" is out of reach.
- Embeddings are English-ish natural language. Identifier search is BM25's job; use
  `--mode keyword` when you already know the name.
- `--max-bytes` skips huge generated files by default (512 KB) — see `lode stats`.

MIT licensed. Weights are governed by their own upstream licenses (bge-small: MIT).
