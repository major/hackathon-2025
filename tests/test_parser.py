"""Tests for markdown parser - DEPRECATED.

This test file tests the old custom parser that has been replaced by python-frontmatter
and Docling. These tests are kept for reference but are currently skipped.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Old parser tests - replaced by python-frontmatter and Docling"
)
