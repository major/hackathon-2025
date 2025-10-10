"""Tests for context expansion and neighbor retrieval. 🔍"""

import pytest

from hackathon.retrieval.context_expansion import ContextExpander


class TestContextExpander:
    """Test ContextExpander class."""

    @pytest.fixture
    def expander(self, db_session):
        """Create a ContextExpander instance."""
        return ContextExpander(db_session)

    def test_context_expander_initialization(self, expander, db_session):
        """Test ContextExpander initialization."""
        assert expander.db is db_session

    def test_get_neighbor_context_basic(self, expander, sample_nodes):
        """Test basic neighbor context retrieval."""
        # Use middle node
        target_node = sample_nodes[2]

        context = expander.get_neighbor_context(target_node, before=1, after=1)

        # Should include before, current, and after
        assert "BEFORE-1" in context
        assert "CURRENT" in context
        assert "AFTER-1" in context
        # Should include actual content
        assert sample_nodes[1].text_content in context
        assert target_node.text_content in context
        assert sample_nodes[3].text_content in context

    def test_get_neighbor_context_multiple_before_after(self, expander, sample_nodes):
        """Test with multiple neighbors in each direction."""
        target_node = sample_nodes[2]

        context = expander.get_neighbor_context(target_node, before=2, after=1)

        assert "BEFORE-2" in context
        assert "BEFORE-1" in context
        assert "CURRENT" in context
        assert "AFTER-1" in context

    def test_get_neighbor_context_at_boundaries(self, expander, sample_nodes):
        """Test context expansion at document boundaries."""
        # First node - no before neighbors
        context = expander.get_neighbor_context(sample_nodes[0], before=2, after=2)
        assert "BEFORE" not in context
        assert "CURRENT" in context
        assert "AFTER" in context

        # Last node - no after neighbors
        context = expander.get_neighbor_context(sample_nodes[-1], before=2, after=2)
        assert "BEFORE" in context
        assert "CURRENT" in context
        assert "AFTER-1" not in context

    def test_get_neighbor_context_formatting(self, expander, sample_nodes):
        """Test that context is properly formatted."""
        target_node = sample_nodes[2]

        context = expander.get_neighbor_context(target_node, before=1, after=1)

        # Should have section headers
        lines = context.split("\n")
        assert any("paragraph" in line for line in lines)  # Node type
        assert any("Section" in line for line in lines)  # Heading metadata

    @pytest.mark.parametrize(
        "before,after",
        [
            (0, 0),
            (1, 0),
            (0, 1),
            (2, 2),
            (5, 5),  # More than available
        ],
    )
    def test_get_neighbor_context_various_counts(
        self, expander, sample_nodes, before, after
    ):
        """Test neighbor context with various before/after counts."""
        target_node = sample_nodes[2]

        context = expander.get_neighbor_context(target_node, before=before, after=after)

        # Should always include current node
        assert "CURRENT" in context or target_node.text_content in context

    def test_get_semantic_block_text_basic(self, expander, sample_nodes):
        """Test getting semantic block text."""
        target_node = sample_nodes[0]

        block_text = expander.get_semantic_block_text(target_node)

        # For flat structure, should just return node text
        assert block_text == target_node.text_content

    def test_get_semantic_block_text_empty_node(self, expander, db_session, sample_document):
        """Test semantic block with node that has no content."""
        from hackathon.models.database import DocumentNode

        empty_node = DocumentNode(
            document_id=sample_document.id,
            node_type="paragraph",
            text_content=None,
            is_leaf=True,
            node_path="chunk_empty",
            position=99,
        )
        db_session.add(empty_node)
        db_session.commit()

        block_text = expander.get_semantic_block_text(empty_node)

        assert block_text == ""

    def test_get_full_document_context(self, expander, sample_nodes):
        """Test getting full document context."""
        target_node = sample_nodes[0]

        full_context = expander.get_full_document_context(target_node)

        # Should include all nodes
        for node in sample_nodes:
            if node.text_content:
                assert node.text_content in full_context

        # Should have document markers
        assert "DOCUMENT" in full_context

    def test_get_full_document_context_ordering(self, expander, sample_nodes):
        """Test that full document context preserves order."""
        target_node = sample_nodes[0]

        full_context = expander.get_full_document_context(target_node)

        # Find positions of each node's text in the full context
        positions = []
        for node in sample_nodes:
            if node.text_content:
                pos = full_context.find(node.text_content)
                assert pos != -1  # Text should be present
                positions.append(pos)

        # Positions should be in ascending order
        assert positions == sorted(positions)

    def test_get_parent_context_level_zero(self, expander, sample_nodes):
        """Test parent context with level 0 (just the node)."""
        target_node = sample_nodes[2]

        context = expander.get_parent_context(target_node, levels=0)

        assert target_node.text_content in context
        assert "CURRENT" in context
        # Should not include neighbors
        assert "BEFORE" not in context
        assert "AFTER" not in context

    def test_get_parent_context_levels(self, expander, sample_nodes):
        """Test parent context with various levels."""
        target_node = sample_nodes[2]

        # Level 1 = 1 before + 1 after
        context = expander.get_parent_context(target_node, levels=1)
        assert sample_nodes[1].text_content in context
        assert target_node.text_content in context
        assert sample_nodes[3].text_content in context

        # Level 2 = 2 before + 2 after
        context = expander.get_parent_context(target_node, levels=2)
        assert sample_nodes[0].text_content in context

    def test_get_parent_context_full_document(self, expander, sample_nodes):
        """Test parent context with level -1 (full document)."""
        target_node = sample_nodes[2]

        context = expander.get_parent_context(target_node, levels=-1)

        # Should be same as full document context
        full_context = expander.get_full_document_context(target_node)
        assert context == full_context

    def test_estimate_context_sizes(self, expander, sample_nodes):
        """Test context size estimation."""
        target_node = sample_nodes[2]

        sizes = expander.estimate_context_sizes(target_node, max_levels=3)

        # Should have various size estimates
        assert "node" in sizes
        assert "level_1" in sizes
        assert "level_2" in sizes
        assert "level_3" in sizes
        assert "full_document" in sizes

        # Sizes should generally increase with levels
        assert sizes["node"] <= sizes["level_1"]
        assert sizes["level_1"] <= sizes["full_document"]

    def test_estimate_context_sizes_values(self, expander, sample_nodes):
        """Test that size estimates are reasonable."""
        target_node = sample_nodes[0]

        sizes = expander.estimate_context_sizes(target_node, max_levels=2)

        # Node size should match text length
        assert sizes["node"] == len(target_node.text_content or "")

        # All sizes should be non-negative
        assert all(size >= 0 for size in sizes.values())

    def test_format_node_content_with_metadata(self, expander, sample_nodes):
        """Test node formatting with metadata."""
        target_node = sample_nodes[0]

        formatted = expander._format_node_content(target_node, "current")

        # Should include node type
        assert target_node.node_type in formatted
        # Should include text content
        assert target_node.text_content in formatted
        # Should have heading if available
        if target_node.meta and target_node.meta.get("headings"):
            assert "Section" in formatted

    def test_format_node_content_no_content(self, expander, db_session, sample_document):
        """Test formatting node with no text content."""
        from hackathon.models.database import DocumentNode

        empty_node = DocumentNode(
            document_id=sample_document.id,
            node_type="paragraph",
            text_content=None,
            is_leaf=True,
            node_path="chunk_empty",
            position=99,
        )

        formatted = expander._format_node_content(empty_node, "current")

        assert "[No content]" in formatted

    @pytest.mark.parametrize(
        "context_type,expected_label",
        [
            ("current", "CURRENT"),
            ("before", "BEFORE"),
            ("after", "AFTER"),
            ("document", "DOCUMENT"),
        ],
    )
    def test_format_node_content_labels(
        self, expander, sample_nodes, context_type, expected_label
    ):
        """Test that format labels are correct."""
        formatted = expander._format_node_content(sample_nodes[0], context_type)
        assert expected_label in formatted

    def test_format_neighbor_node_basic(self, expander, sample_nodes):
        """Test neighbor node formatting."""
        node = sample_nodes[1]

        formatted = expander._format_neighbor_node(node, "BEFORE-1")

        assert "BEFORE-1" in formatted
        assert node.node_type in formatted
        assert node.text_content in formatted
        # Should include position
        assert "Position: 1" in formatted

    def test_format_neighbor_node_with_headings(self, expander, sample_nodes):
        """Test neighbor formatting includes heading metadata."""
        node = sample_nodes[2]

        formatted = expander._format_neighbor_node(node, "AFTER-1")

        # Should include heading if available
        if node.meta and node.meta.get("headings"):
            assert "Section" in formatted

    def test_format_neighbor_node_no_content(self, expander, db_session, sample_document):
        """Test neighbor formatting with no content."""
        from hackathon.models.database import DocumentNode

        empty_node = DocumentNode(
            document_id=sample_document.id,
            node_type="paragraph",
            text_content=None,
            is_leaf=True,
            node_path="chunk_empty",
            position=50,
        )

        formatted = expander._format_neighbor_node(empty_node, "AFTER-2")

        assert "[No content]" in formatted
        assert "AFTER-2" in formatted

    def test_context_expansion_with_code_blocks(
        self, expander, db_session, sample_document
    ):
        """Test context expansion with code block nodes."""
        from hackathon.models.database import DocumentNode

        # Create nodes with code
        code_node = DocumentNode(
            document_id=sample_document.id,
            node_type="code",
            text_content="```python\nprint('hello')\n```",
            is_leaf=True,
            node_path="chunk_code",
            position=10,
            meta={"language": "python"},
        )
        db_session.add(code_node)
        db_session.commit()

        context = expander.get_neighbor_context(code_node, before=0, after=0)

        assert "code" in context  # Node type
        assert "print('hello')" in context  # Code content

    def test_context_expansion_performance(self, expander, db_session, sample_document):
        """Test that context expansion performs reasonably with many nodes."""
        from hackathon.models.database import DocumentNode

        # Create many nodes
        nodes = []
        for i in range(50):
            node = DocumentNode(
                document_id=sample_document.id,
                node_type="paragraph",
                text_content=f"Content {i}",
                is_leaf=True,
                node_path=f"chunk_{i}",
                position=i,
                meta={"headings": f"Section {i // 10}"},
            )
            nodes.append(node)
        db_session.add_all(nodes)
        db_session.commit()

        # Get context for middle node - should complete quickly
        target_node = nodes[25]
        context = expander.get_neighbor_context(target_node, before=5, after=5)

        # Should include expected neighbors
        assert "Content 20" in context  # 5 before
        assert "Content 25" in context  # Current
        assert "Content 30" in context  # 5 after

    def test_get_parent_context_ignore_siblings_param(self, expander, sample_nodes):
        """Test that _include_siblings parameter is ignored (API compatibility)."""
        target_node = sample_nodes[2]

        # Should work the same regardless of _include_siblings value
        context1 = expander.get_parent_context(target_node, levels=1, _include_siblings=True)
        context2 = expander.get_parent_context(target_node, levels=1, _include_siblings=False)

        # In flat structure, both should be the same
        assert context1 == context2
