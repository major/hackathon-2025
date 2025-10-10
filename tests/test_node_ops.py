"""Tests for document node operations. 🗂️"""

import pytest

from hackathon.database.node_ops import (
    create_document_node,
    get_all_leaf_nodes,
    get_neighbors,
    get_neighbors_after,
    get_neighbors_before,
)
from hackathon.models.database import DocumentNode
from hackathon.models.schemas import DocumentNodeCreate


class TestCreateDocumentNode:
    """Test document node creation."""

    def test_create_node_basic(self, db_session, sample_document):
        """Test creating a basic document node."""
        node_data = DocumentNodeCreate(
            document_id=sample_document.id,
            node_type="paragraph",
            text_content="Test content",
            is_leaf=True,
            node_path="chunk_0",
            position=0,
        )

        node = create_document_node(db_session, node_data)

        assert node.id is not None
        assert node.document_id == sample_document.id
        assert node.node_type == "paragraph"
        assert node.text_content == "Test content"
        assert node.is_leaf is True
        assert node.position == 0

    def test_create_node_with_metadata(self, db_session, sample_document):
        """Test creating node with metadata."""
        node_data = DocumentNodeCreate(
            document_id=sample_document.id,
            node_type="code",
            text_content="print('hello')",
            is_leaf=True,
            node_path="chunk_0",
            position=0,
            metadata={"language": "python", "headings": "Code Examples > Python"},
        )

        node = create_document_node(db_session, node_data)

        assert node.meta is not None
        assert node.meta["language"] == "python"
        assert node.meta["headings"] == "Code Examples > Python"

    @pytest.mark.parametrize(
        "node_type,text_content,position",
        [
            ("paragraph", "First paragraph", 0),
            ("code", "```python\ncode\n```", 1),
            ("list", "- Item 1\n- Item 2", 2),
            ("section_header", "# Heading", 3),
        ],
    )
    def test_create_node_various_types(
        self, db_session, sample_document, node_type, text_content, position
    ):
        """Test creating nodes with various types."""
        node_data = DocumentNodeCreate(
            document_id=sample_document.id,
            node_type=node_type,
            text_content=text_content,
            is_leaf=True,
            node_path=f"chunk_{position}",
            position=position,
        )

        node = create_document_node(db_session, node_data)

        assert node.node_type == node_type
        assert node.text_content == text_content
        assert node.position == position


class TestGetAllLeafNodes:
    """Test retrieving all leaf nodes."""

    def test_get_all_leaf_nodes_single_document(self, db_session, sample_nodes):
        """Test getting all leaf nodes for a single document."""
        document_id = sample_nodes[0].document_id
        nodes = get_all_leaf_nodes(db_session, document_id)

        assert len(nodes) == len(sample_nodes)
        assert all(n.is_leaf for n in nodes)
        assert all(n.document_id == document_id for n in nodes)

    def test_get_all_leaf_nodes_no_filter(self, db_session, sample_nodes):
        """Test getting all leaf nodes without document filter."""
        nodes = get_all_leaf_nodes(db_session)

        assert len(nodes) >= len(sample_nodes)
        assert all(n.is_leaf for n in nodes)

    def test_get_all_leaf_nodes_empty_document(self, db_session, sample_document):
        """Test getting nodes when document has no nodes."""
        # Create a new document without nodes
        from hackathon.models.database import Document

        empty_doc = Document(filename="empty.md", filepath="/path/empty.md")
        db_session.add(empty_doc)
        db_session.commit()

        nodes = get_all_leaf_nodes(db_session, empty_doc.id)

        assert nodes == []

    def test_get_all_leaf_nodes_non_leaf_nodes(
        self, db_session, sample_document
    ):
        """Test that non-leaf nodes are excluded."""
        # Create non-leaf node
        non_leaf = DocumentNode(
            document_id=sample_document.id,
            node_type="section",
            text_content="Section",
            is_leaf=False,
            node_path="section_0",
            position=99,
        )
        db_session.add(non_leaf)
        db_session.commit()

        nodes = get_all_leaf_nodes(db_session, sample_document.id)

        # Should not include non-leaf node
        assert all(n.is_leaf for n in nodes)
        assert non_leaf.id not in [n.id for n in nodes]


class TestGetNeighborsBefore:
    """Test retrieving nodes before a given position."""

    def test_get_neighbors_before_basic(self, db_session, sample_nodes):
        """Test getting neighbors before a node."""
        # Get node at position 2
        target_node = sample_nodes[2]

        # Get 2 neighbors before
        neighbors = get_neighbors_before(db_session, target_node, count=2)

        assert len(neighbors) == 2
        # Should be in ascending order (oldest to newest)
        assert neighbors[0].position == 0
        assert neighbors[1].position == 1

    def test_get_neighbors_before_at_start(self, db_session, sample_nodes):
        """Test getting neighbors when at start of document."""
        # Node at position 0 has no previous neighbors
        target_node = sample_nodes[0]

        neighbors = get_neighbors_before(db_session, target_node, count=2)

        assert neighbors == []

    def test_get_neighbors_before_limited_availability(self, db_session, sample_nodes):
        """Test when fewer neighbors exist than requested."""
        # Node at position 1 has only 1 neighbor before
        target_node = sample_nodes[1]

        neighbors = get_neighbors_before(db_session, target_node, count=5)

        assert len(neighbors) == 1
        assert neighbors[0].position == 0

    def test_get_neighbors_before_count_variations(self, db_session, sample_nodes):
        """Test with various count values."""
        target_node = sample_nodes[3]  # Last node

        # Count = 1
        neighbors = get_neighbors_before(db_session, target_node, count=1)
        assert len(neighbors) == 1
        assert neighbors[0].position == 2

        # Count = 2
        neighbors = get_neighbors_before(db_session, target_node, count=2)
        assert len(neighbors) == 2
        assert neighbors[0].position == 1
        assert neighbors[1].position == 2

    def test_get_neighbors_before_no_position(self, db_session, sample_document):
        """Test behavior when node has no position set."""
        # Create node without position
        node = DocumentNode(
            document_id=sample_document.id,
            node_type="paragraph",
            text_content="No position",
            is_leaf=True,
            node_path="chunk_x",
            position=None,
        )
        db_session.add(node)
        db_session.commit()

        neighbors = get_neighbors_before(db_session, node, count=2)

        assert neighbors == []


class TestGetNeighborsAfter:
    """Test retrieving nodes after a given position."""

    def test_get_neighbors_after_basic(self, db_session, sample_nodes):
        """Test getting neighbors after a node."""
        # Get node at position 1
        target_node = sample_nodes[1]

        # Get 2 neighbors after
        neighbors = get_neighbors_after(db_session, target_node, count=2)

        assert len(neighbors) == 2
        # Should be in ascending order (closest to furthest)
        assert neighbors[0].position == 2
        assert neighbors[1].position == 3

    def test_get_neighbors_after_at_end(self, db_session, sample_nodes):
        """Test getting neighbors when at end of document."""
        # Last node has no next neighbors
        target_node = sample_nodes[-1]

        neighbors = get_neighbors_after(db_session, target_node, count=2)

        assert neighbors == []

    def test_get_neighbors_after_limited_availability(self, db_session, sample_nodes):
        """Test when fewer neighbors exist than requested."""
        # Second-to-last node has only 1 neighbor after
        target_node = sample_nodes[-2]

        neighbors = get_neighbors_after(db_session, target_node, count=5)

        assert len(neighbors) == 1
        assert neighbors[0].position == sample_nodes[-1].position

    def test_get_neighbors_after_count_variations(self, db_session, sample_nodes):
        """Test with various count values."""
        target_node = sample_nodes[0]  # First node

        # Count = 1
        neighbors = get_neighbors_after(db_session, target_node, count=1)
        assert len(neighbors) == 1
        assert neighbors[0].position == 1

        # Count = 3
        neighbors = get_neighbors_after(db_session, target_node, count=3)
        assert len(neighbors) == 3
        assert neighbors[0].position == 1
        assert neighbors[1].position == 2
        assert neighbors[2].position == 3

    def test_get_neighbors_after_no_position(self, db_session, sample_document):
        """Test behavior when node has no position set."""
        node = DocumentNode(
            document_id=sample_document.id,
            node_type="paragraph",
            text_content="No position",
            is_leaf=True,
            node_path="chunk_x",
            position=None,
        )
        db_session.add(node)
        db_session.commit()

        neighbors = get_neighbors_after(db_session, node, count=2)

        assert neighbors == []


class TestGetNeighbors:
    """Test retrieving neighbors before and after."""

    def test_get_neighbors_both_directions(self, db_session, sample_nodes):
        """Test getting neighbors in both directions."""
        # Middle node
        target_node = sample_nodes[2]

        before, after = get_neighbors(db_session, target_node, before=2, after=1)

        assert len(before) == 2
        assert len(after) == 1
        # Before: positions 0, 1
        assert before[0].position == 0
        assert before[1].position == 1
        # After: position 3
        assert after[0].position == 3

    def test_get_neighbors_symmetric(self, db_session, sample_nodes):
        """Test getting same number of neighbors before and after."""
        target_node = sample_nodes[2]

        before, after = get_neighbors(db_session, target_node, before=1, after=1)

        assert len(before) == 1
        assert len(after) == 1
        assert before[0].position == 1
        assert after[0].position == 3

    def test_get_neighbors_at_document_boundaries(self, db_session, sample_nodes):
        """Test neighbors at document boundaries."""
        # First node
        before, after = get_neighbors(db_session, sample_nodes[0], before=2, after=2)
        assert len(before) == 0
        assert len(after) == 2

        # Last node
        before, after = get_neighbors(db_session, sample_nodes[-1], before=2, after=2)
        assert len(before) == 2
        assert len(after) == 0

    @pytest.mark.parametrize(
        "position,before_count,after_count,expected_before,expected_after",
        [
            (1, 1, 1, 1, 1),  # Middle with symmetric
            (0, 2, 2, 0, 2),  # Start of document
            (3, 2, 2, 2, 0),  # End of document (assuming 4 nodes)
            (2, 0, 0, 0, 0),  # No neighbors requested
            (1, 5, 5, 1, 2),  # Request more than available
        ],
    )
    def test_get_neighbors_various_scenarios(
        self,
        db_session,
        sample_nodes,
        position,
        before_count,
        after_count,
        expected_before,
        expected_after,
    ):
        """Test various neighbor retrieval scenarios."""
        target_node = sample_nodes[position]

        before, after = get_neighbors(
            db_session, target_node, before=before_count, after=after_count
        )

        assert len(before) == expected_before
        assert len(after) == expected_after

    def test_get_neighbors_ordering(self, db_session, sample_nodes):
        """Test that neighbors are properly ordered."""
        target_node = sample_nodes[2]

        before, after = get_neighbors(db_session, target_node, before=2, after=1)

        # Before should be oldest to newest (ascending positions)
        if len(before) > 1:
            for i in range(len(before) - 1):
                assert before[i].position < before[i + 1].position

        # After should be closest to furthest (ascending positions)
        if len(after) > 1:
            for i in range(len(after) - 1):
                assert after[i].position < after[i + 1].position

    def test_get_neighbors_different_documents(
        self, db_session, sample_document
    ):
        """Test that neighbors are scoped to the same document."""
        # Create second document with nodes
        from hackathon.models.database import Document

        doc2 = Document(filename="doc2.md", filepath="/path/doc2.md")
        db_session.add(doc2)
        db_session.commit()

        # Add nodes to second document
        for i in range(3):
            node = DocumentNode(
                document_id=doc2.id,
                node_type="paragraph",
                text_content=f"Doc2 content {i}",
                is_leaf=True,
                node_path=f"chunk_{i}",
                position=i,
            )
            db_session.add(node)
        db_session.commit()

        # Get nodes from first document
        doc1_nodes = get_all_leaf_nodes(db_session, sample_document.id)
        if doc1_nodes:
            target = doc1_nodes[0]
            before, after = get_neighbors(db_session, target, before=5, after=5)

            # All neighbors should be from same document
            assert all(n.document_id == sample_document.id for n in before)
            assert all(n.document_id == sample_document.id for n in after)
