"""Lexical analysis is what makes identifier queries work."""

from __future__ import annotations

from lode import lexicon


def test_camel_case_is_indexed_whole_and_split():
    terms = lexicon.terms("HttpAuthMiddleware")
    assert "httpauthmiddleware" in terms
    assert {"http", "auth", "middleware"} <= set(terms)


def test_acronym_boundaries():
    assert {"http", "server"} <= set(lexicon.terms("HTTPServer"))


def test_snake_and_kebab():
    assert {"parse", "url"} <= set(lexicon.terms("parse_url"))
    assert {"parse", "url"} <= set(lexicon.terms("parse-url"))


def test_stopwords_and_short_tokens_are_dropped():
    terms = lexicon.terms("the result of a b 42 x2")
    assert "the" not in terms and "of" not in terms
    assert "42" not in terms
    assert "b" not in terms, "single characters carry no signal"


def test_dotted_module_paths_survive():
    terms = lexicon.terms("lode.search_engine")
    assert "search" in terms and "engine" in terms


def test_query_terms_are_unique_and_ordered():
    assert lexicon.query_terms("auth auth token") == ["auth", "token"]


def test_field_boosts_weight_path_and_symbol():
    counts = lexicon.lexicon("src/auth/login.py", "check_password", "def", "return 1")
    assert counts["auth"] >= 2 * counts["return"], "path terms should outweigh body terms"
    assert counts["check"] >= 2


def test_counts_is_a_counter_over_terms():
    counts = lexicon.counts("alpha beta alpha")
    assert counts["alpha"] == 2 and counts["beta"] == 1


def test_merge_requires_matching_weights():
    import pytest

    with pytest.raises(ValueError):
        lexicon.merge(lexicon.counts("a"), weights=(1, 2))
