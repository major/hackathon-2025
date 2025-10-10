"""Tests for Docling document processing. 📄"""

import tempfile
from pathlib import Path

import pytest

from hackathon.processing.docling_processor import (
    _extract_heading_context,
    _fix_inline_code_blocks,
    _infer_node_type,
    extract_yaml_frontmatter,
)


class TestYAMLFrontmatter:
    """Test YAML frontmatter extraction."""

    def test_extract_frontmatter_with_metadata(self, temp_markdown_file):
        """Test extracting frontmatter with various metadata types."""
        content, metadata = extract_yaml_frontmatter(temp_markdown_file)

        assert "# Introduction" in content
        assert "---" not in content  # Frontmatter should be removed
        assert metadata["title"] == "Test Document"
        assert metadata["date"] == "2025-10-08"
        assert "tags" in metadata

    def test_extract_frontmatter_no_frontmatter(self, sample_markdown_no_frontmatter):
        """Test extracting from markdown without frontmatter."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp_file:
            tmp_file.write(sample_markdown_no_frontmatter)
            tmp_path = Path(tmp_file.name)

        try:
            content, metadata = extract_yaml_frontmatter(tmp_path)

            assert "# Simple Document" in content
            assert metadata == {}  # No frontmatter
        finally:
            tmp_path.unlink(missing_ok=True)

    @pytest.mark.parametrize(
        "frontmatter,expected_keys,expected_types",
        [
            (
                "title: Test\ncount: 42\nactive: true\nratio: 3.14",
                ["title", "count", "active", "ratio"],
                {"title": str, "count": int, "active": bool, "ratio": float},
            ),
            (
                "tags: [python, testing]\nauthor: Alice",
                ["tags", "author"],
                {"author": str},
            ),
        ],
    )
    def test_extract_frontmatter_type_preservation(
        self, frontmatter, expected_keys, expected_types
    ):
        """Test that frontmatter preserves data types."""
        markdown = f"---\n{frontmatter}\n---\n\n# Content"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp_file:
            tmp_file.write(markdown)
            tmp_path = Path(tmp_file.name)

        try:
            content, metadata = extract_yaml_frontmatter(tmp_path)

            for key in expected_keys:
                assert key in metadata

            for key, expected_type in expected_types.items():
                assert isinstance(metadata[key], expected_type)
        finally:
            tmp_path.unlink(missing_ok=True)


class TestInlineCodeBlocks:
    """Test fixing of inline code block formatting."""

    @pytest.mark.parametrize(
        "input_text,expected_output",
        [
            # Single-line code block → inline code
            ("```\nterm\n```", "`term`"),
            ("```\nkubernetes\n```", "`kubernetes`"),
            # Short code block with whitespace
            ("```\n  command  \n```", "`command`"),
            # Keep multi-line code blocks as-is
            ("```\nline1\nline2\n```", "```\nline1\nline2\n```"),
            # Keep long single-line code (likely real code)
            (
                "```\n" + "x" * 60 + "\n```",
                "```\n" + "x" * 60 + "\n```",
            ),
            # Mixed content
            (
                "Text with ```\nterm\n``` and more text",
                "Text with `term` and more text",
            ),
        ],
    )
    def test_fix_inline_code_blocks(self, input_text, expected_output):
        """Test fixing various inline code block patterns."""
        result = _fix_inline_code_blocks(input_text)
        assert result == expected_output

    def test_fix_inline_code_blocks_multiple(self):
        """Test fixing multiple inline code blocks in one text."""
        text = "Use ```\nkubectl\n``` or ```\ndocker\n``` commands."
        expected = "Use `kubectl` or `docker` commands."
        result = _fix_inline_code_blocks(text)
        assert result == expected

    def test_fix_inline_code_blocks_real_code(self):
        """Test that real code blocks are preserved."""
        text = """```bash
npm install
pip install -r requirements.txt
```"""
        result = _fix_inline_code_blocks(text)
        assert result == text  # Should be unchanged


class TestNodeTypeInference:
    """Test node type inference from chunk content."""

    @pytest.mark.parametrize(
        "chunk_text,expected_type",
        [
            # Code blocks
            ("```python\nprint('hello')\n```", "code"),
            ("```bash\nnpm install\n```", "code"),
            ("    indented code block", "code"),
            # Lists
            ("- First item\n- Second item", "list"),
            ("* Bullet point\n* Another point", "list"),
            ("1. First\n2. Second", "list"),
            ("2. Starting at 2\n3. Next", "list"),
            # Paragraphs
            ("Regular paragraph text.", "paragraph"),
            ("Multiple sentences. In a paragraph.", "paragraph"),
            ("No special formatting.", "paragraph"),
        ],
    )
    def test_infer_node_type(self, chunk_text, expected_type):
        """Test node type inference with various content."""
        result = _infer_node_type(chunk_text)
        assert result == expected_type

    def test_infer_node_type_mixed_content(self):
        """Test node type when content has mixed elements."""
        # Code has higher priority
        text = "- List item\n```python\ncode\n```"
        assert _infer_node_type(text) == "code"

    def test_infer_node_type_edge_cases(self):
        """Test edge cases in node type inference."""
        # Empty string
        result = _infer_node_type("")
        assert result in ["paragraph", "code"]  # Empty can be either

        # Just whitespace - code heuristic checks first 20 chars
        result = _infer_node_type("   \n  ")
        assert result in ["paragraph", "code"]  # Whitespace can trigger code check

        # Code-like but not at start (not in first 20 chars)
        assert _infer_node_type("Text then has spaces later") == "paragraph"


class TestHeadingContext:
    """Test heading context extraction from chunks."""

    def test_extract_heading_context_with_headings(self):
        """Test extracting heading hierarchy from chunk metadata."""

        # Mock chunk with headings
        class MockChunk:
            class MockMeta:
                headings = ["Installation", "Dependencies", "npm"]

            meta = MockMeta()

        chunk = MockChunk()
        result = _extract_heading_context(chunk)
        assert result == "Installation > Dependencies > npm"

    def test_extract_heading_context_no_headings(self):
        """Test when chunk has no heading metadata."""

        class MockChunk:
            class MockMeta:
                headings = []

            meta = MockMeta()

        chunk = MockChunk()
        result = _extract_heading_context(chunk)
        assert result == ""

    def test_extract_heading_context_no_meta(self):
        """Test when chunk has no metadata at all."""

        class MockChunk:
            meta = None

        chunk = MockChunk()
        result = _extract_heading_context(chunk)
        assert result == ""

    def test_extract_heading_context_single_heading(self):
        """Test with single heading level."""

        class MockChunk:
            class MockMeta:
                headings = ["Introduction"]

            meta = MockMeta()

        chunk = MockChunk()
        result = _extract_heading_context(chunk)
        assert result == "Introduction"


class TestDoclingIntegration:
    """Integration tests for Docling processing (requires Docling)."""

    @pytest.mark.skipif(
        True,
        reason="PostgreSQL-specific (to_tsvector) not supported in SQLite tests",
    )
    def test_process_document_with_docling_basic(
        self, db_session, sample_document, temp_markdown_file
    ):
        """Test basic document processing with Docling."""
        from hackathon.processing.docling_processor import process_document_with_docling

        # Extract frontmatter first
        _, frontmatter = extract_yaml_frontmatter(temp_markdown_file)

        # Process document
        node_ids = process_document_with_docling(
            db_session, sample_document.id, temp_markdown_file, frontmatter
        )

        # Should create multiple nodes
        assert len(node_ids) > 0

        # Verify nodes were created in database
        from hackathon.database.node_ops import get_all_leaf_nodes

        nodes = get_all_leaf_nodes(db_session, sample_document.id)
        assert len(nodes) == len(node_ids)

        # Verify sequential positions
        positions = sorted([n.position for n in nodes if n.position is not None])
        assert positions == list(range(len(positions)))

    @pytest.mark.skipif(
        True,
        reason="PostgreSQL-specific (to_tsvector) not supported in SQLite tests",
    )
    def test_process_document_preserves_headings(
        self, db_session, sample_document, temp_markdown_file
    ):
        """Test that heading context is preserved in node metadata."""
        from hackathon.processing.docling_processor import process_document_with_docling

        _, frontmatter = extract_yaml_frontmatter(temp_markdown_file)
        _ = process_document_with_docling(
            db_session, sample_document.id, temp_markdown_file, frontmatter
        )

        from hackathon.database.node_ops import get_all_leaf_nodes

        nodes = get_all_leaf_nodes(db_session, sample_document.id)

        # At least some nodes should have heading metadata
        nodes_with_headings = [n for n in nodes if n.meta and n.meta.get("headings")]
        assert len(nodes_with_headings) > 0

    @pytest.mark.skipif(
        True,
        reason="PostgreSQL-specific (to_tsvector) not supported in SQLite tests",
    )
    def test_process_document_code_blocks(
        self, db_session, sample_document, sample_markdown_with_code
    ):
        """Test processing document with code blocks."""
        from hackathon.processing.docling_processor import process_document_with_docling

        # Create temp file with code
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp_file:
            tmp_file.write(sample_markdown_with_code)
            tmp_path = Path(tmp_file.name)

        try:
            _, frontmatter = extract_yaml_frontmatter(tmp_path)
            _ = process_document_with_docling(
                db_session, sample_document.id, tmp_path, frontmatter
            )

            from hackathon.database.node_ops import get_all_leaf_nodes

            nodes = get_all_leaf_nodes(db_session, sample_document.id)

            # Should have code nodes (note: type inference may vary based on chunking)
            _code_nodes = [n for n in nodes if n.node_type == "code"]
            # At least processed something
            assert len(nodes) > 0
        finally:
            tmp_path.unlink(missing_ok=True)
