"""Cache bookkeeping: what "ready" is allowed to mean, and how mirrors get picked.

No network here: the download loop is exercised through its file list and readiness
predicate, not by fetching weights.
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from lode import modelsource


def test_a_flaky_mirror_is_retried_until_it_answers(monkeypatch, tmp_path):
    """Small side-car requests get reset often enough that one try is not a plan."""
    attempts = []
    slept = []

    class Response:
        """Answers once with three bytes, then EOF, like a real body would."""

        headers = {"Content-Length": "3"}

        def __init__(self) -> None:
            self.body = [b"abc"]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, _size):
            return self.body.pop(0) if self.body else b""

    def fake_urlopen(request, timeout=None):
        attempts.append(request.full_url)
        if len(attempts) < 3:
            raise urllib.error.URLError("connection reset by peer")
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(modelsource.time, "sleep", lambda seconds: slept.append(seconds))

    dest = tmp_path / "tokenizer.json"
    modelsource._fetch("https://example.test/tokenizer.json", dest)
    assert dest.read_bytes() == b"abc"
    assert len(attempts) == 3
    assert len(slept) == 2, "backoff between attempts, not a busy loop"


def test_giving_up_names_the_url_that_failed(monkeypatch, tmp_path):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection reset by peer")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(modelsource.time, "sleep", lambda seconds: None)

    with pytest.raises(modelsource.ModelUnavailable) as caught:
        modelsource._fetch("https://example.test/tokenizer.json", tmp_path / "nope.json")
    assert "https://example.test/tokenizer.json" in str(caught.value)


def test_a_truncated_body_is_not_accepted_as_a_weight_file(monkeypatch, tmp_path):
    """Content-Length is a contract, and a partial file is not a usable tokenizer.

    The mirror used to stop mid-body and lode renamed whatever arrived into the cache, so
    the index looked ready and embedded nothing.
    """
    slept = []

    class ShortResponse:
        headers = {"Content-Length": "100"}

        def __init__(self) -> None:
            self.left = 10

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, _size):
            block = b"x" * min(_size, self.left)
            self.left -= len(block)
            return block

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=None: ShortResponse())
    monkeypatch.setattr(modelsource.time, "sleep", lambda seconds: slept.append(seconds))

    dest = tmp_path / "tokenizer.json"
    with pytest.raises(modelsource.ModelUnavailable) as caught:
        modelsource._fetch("https://example.test/tokenizer.json", dest)
    assert "10 of 100" in str(caught.value)
    assert not dest.exists(), "a half file must not land in the cache"
    assert slept, "a short read is retried before it is reported"


def test_the_tokenizer_config_counts_as_required_not_optional():
    """A cache with weights but no tokenizer_config.json is not usable.

    fastembed raises "Could not find tokenizer_config.json" and lode then indexes the
    whole repo keyword-only while reporting the weights as ready, so readiness has to
    include the file the tokenizer refuses to start without.
    """
    assert "tokenizer_config.json" in modelsource._SIDE_CAR_REQUIRED
    assert "tokenizer_config.json" not in modelsource._SIDE_CAR_OPTIONAL


def _spec(**overrides) -> modelsource.ModelSpec:
    fields = dict(
        alias="BAAI/example",
        hf_repo="BAAI/example",
        revision="main",
        weight_file="onnx/model.onnx",
        required=(
            "onnx/model.onnx",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
        optional=("modules.json",),
        dim=384,
        max_tokens=512,
        note="test",
    )
    fields.update(overrides)
    return modelsource.ModelSpec(**fields)


def _write(directory, *names) -> None:
    """Weight files live under onnx/, so the parent has to exist too."""
    for name in names:
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")


def _ready_cache(spec, directory) -> None:
    _write(directory, *spec.required)
    _write(directory, modelsource.MANIFEST)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    def build(spec):
        directory = tmp_path / spec.dir_name
        directory.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(modelsource, "model_dir", lambda s: directory)
        return directory

    return build


def test_a_full_cache_is_ready(cache):
    spec = _spec()
    _ready_cache(spec, cache(spec))
    assert modelsource.is_ready(spec)


def test_weights_without_a_tokenizer_config_are_not_ready(cache):
    spec = _spec()
    directory = cache(spec)
    _write(directory, *[n for n in spec.required if n != "tokenizer_config.json"])
    _write(directory, modelsource.MANIFEST)
    assert not modelsource.is_ready(spec), "this cache downloaded fine and still cannot embed"


def test_an_absent_optional_file_does_not_block_readiness(cache):
    spec = _spec()
    directory = cache(spec)
    _ready_cache(spec, directory)
    assert not (directory / "modules.json").exists()
    assert modelsource.is_ready(spec)


def test_no_manifest_never_means_ready(cache):
    spec = _spec()
    directory = cache(spec)
    _write(directory, *spec.required)
    assert not modelsource.is_ready(spec)


def test_mirrors_default_to_the_one_that_reaches_the_same_tree(monkeypatch):
    monkeypatch.delenv("LODE_MIRRORS", raising=False)
    mirrors = modelsource.default_mirrors()
    assert len(mirrors) >= 2
    assert all("{repo}" in m and "{rev}" in m and "{path}" in m for m in mirrors)


def test_lode_mirrors_override_the_chain(monkeypatch):
    monkeypatch.setenv("LODE_MIRRORS", "https://a.example/{repo}/{rev}/{path}, https://b.example/{repo}/{rev}/{path}")
    assert modelsource.default_mirrors() == [
        "https://a.example/{repo}/{rev}/{path}",
        "https://b.example/{repo}/{rev}/{path}",
    ]


def test_unknown_model_is_an_explained_error():
    with pytest.raises(modelsource.ModelUnavailable):
        modelsource.resolve_spec("definitely/not-a-model")
