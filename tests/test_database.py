"""Tests for database operations."""

from hackathon.database.document_ops import (
    create_document,
    get_document_by_filename,
)
from hackathon.database.node_ops import create_document_node
from hackathon.models.schemas import DocumentCreate, DocumentNodeCreate


def test_create_document(db_session):
    """Test document creation."""
    doc_data = DocumentCreate(
        filename="test.md", filepath="/path/to/test.md", metadata={"key": "value"}
    )

    doc = create_document(db_session, doc_data)

    assert doc.id is not None
    assert doc.filename == "test.md"
    assert doc.filepath == "/path/to/test.md"
    assert doc.meta == {"key": "value"}


def test_get_document_by_filename(db_session):
    """Test retrieving document by filename."""
    doc_data = DocumentCreate(filename="test.md", filepath="/path/to/test.md")
    created_doc = create_document(db_session, doc_data)

    retrieved_doc = get_document_by_filename(db_session, "test.md")

    assert retrieved_doc is not None
    assert retrieved_doc.id == created_doc.id


def test_create_document_node(db_session):
    """Test document node creation with flat structure."""
    # Create document first
    doc_data = DocumentCreate(filename="test.md", filepath="/path/to/test.md")
    doc = create_document(db_session, doc_data)

    # Create node (flat structure - no parent_id)
    node_data = DocumentNodeCreate(
        document_id=doc.id,
        node_type="paragraph",
        text_content="Test paragraph",
        is_leaf=True,
        node_path="chunk_0",
        position=0,
    )

    node = create_document_node(db_session, node_data)

    assert node.id is not None
    assert node.document_id == doc.id
    assert node.node_type == "paragraph"
    assert node.is_leaf is True
    assert node.position == 0


def test_sequential_nodes(db_session):
    """Test creating sequential nodes in flat structure."""
    # Create document
    doc_data = DocumentCreate(filename="test.md", filepath="/path/to/test.md")
    doc = create_document(db_session, doc_data)

    # Create multiple sequential nodes
    nodes = []
    for i in range(3):
        node_data = DocumentNodeCreate(
            document_id=doc.id,
            node_type="paragraph",
            text_content=f"Paragraph {i}",
            is_leaf=True,
            node_path=f"chunk_{i}",
            position=i,
        )
        node = create_document_node(db_session, node_data)
        nodes.append(node)

    # Verify sequential positions
    assert nodes[0].position == 0
    assert nodes[1].position == 1
    assert nodes[2].position == 2

    # Verify all belong to same document
    for node in nodes:
        assert node.document_id == doc.id


def test_node_with_metadata(db_session):
    """Test node creation with metadata."""
    doc_data = DocumentCreate(filename="test.md", filepath="/path/to/test.md")
    doc = create_document(db_session, doc_data)

    node_data = DocumentNodeCreate(
        document_id=doc.id,
        node_type="code",
        text_content="print('hello')",
        is_leaf=True,
        node_path="chunk_0",
        position=0,
        metadata={"language": "python", "headings": "Introduction > Setup"},
    )

    node = create_document_node(db_session, node_data)

    assert node.id is not None
    assert node.meta == {"language": "python", "headings": "Introduction > Setup"}
