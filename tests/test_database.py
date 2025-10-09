"""Tests for database operations."""

from hackathon.database.operations import (
    create_document,
    create_document_node,
    get_document_by_filename,
    get_node_ancestors,
)
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
    """Test document node creation."""
    # Create document first
    doc_data = DocumentCreate(filename="test.md", filepath="/path/to/test.md")
    doc = create_document(db_session, doc_data)

    # Create node
    node_data = DocumentNodeCreate(
        document_id=doc.id,
        parent_id=None,
        node_type="section",
        text_content="Test section",
        is_leaf=False,
        node_path="0",
    )

    node = create_document_node(db_session, node_data)

    assert node.id is not None
    assert node.document_id == doc.id
    assert node.node_type == "section"
    assert node.is_leaf is False


def test_node_hierarchy(db_session):
    """Test hierarchical node relationships."""
    # Create document
    doc_data = DocumentCreate(filename="test.md", filepath="/path/to/test.md")
    doc = create_document(db_session, doc_data)

    # Create parent node
    parent_data = DocumentNodeCreate(
        document_id=doc.id,
        parent_id=None,
        node_type="section",
        text_content="Parent section",
        is_leaf=False,
        node_path="0",
    )
    parent = create_document_node(db_session, parent_data)

    # Create child node
    child_data = DocumentNodeCreate(
        document_id=doc.id,
        parent_id=parent.id,
        node_type="paragraph",
        text_content="Child paragraph",
        is_leaf=True,
        node_path="0.0",
    )
    child = create_document_node(db_session, child_data)

    # Test relationship
    assert child.parent_id == parent.id

    # Test ancestor retrieval
    ancestors = get_node_ancestors(db_session, child.id)
    assert len(ancestors) == 1
    assert ancestors[0].id == parent.id
