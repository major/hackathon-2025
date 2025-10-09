"""Tests for BM25 indexing."""

import pytest

from hackathon.processing.bm25 import tokenize_text


@pytest.mark.parametrize(
    ("text", "expected_tokens"),
    [
        ("Hello World", ["hello", "world"]),
        ("Test-Document with-hyphens", ["test", "document", "with", "hyphens"]),
        ("CamelCase and spaces", ["camelcase", "and", "spaces"]),
        ("Numbers 123 and 456", ["numbers", "123", "and", "456"]),
    ],
)
def test_tokenize_text(text, expected_tokens):
    """Test text tokenization."""
    tokens = tokenize_text(text)

    assert tokens == expected_tokens


def test_tokenize_filters_short_tokens():
    """Test that single-character tokens are filtered."""
    text = "a b cd efg"
    tokens = tokenize_text(text)

    assert "a" not in tokens
    assert "b" not in tokens
    assert "cd" in tokens
    assert "efg" in tokens
