"""Database CRUD operations."""

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from hackathon.models.database import (
    BM25Index,
    ContextualChunk,
    Document,
    DocumentNode,
    Embedding,
)
from hackathon.models.schemas import (
    BM25IndexCreate,
    ContextualChunkCreate,
    DocumentCreate,
    DocumentNodeCreate,
    EmbeddingCreate,
)


def create_document(db: Session, doc: DocumentCreate) -> Document:
    """
    Create a new document in the database, or update if it already exists (upsert).
    When updating, deletes all existing nodes to avoid constraint violations.

    Args:
        db: Database session
        doc: Document data to create

    Returns:
        Created or updated Document instance
    """
    doc_dict = doc.model_dump()
    # Map 'metadata' from schema to 'meta' in database model
    if "metadata" in doc_dict:
        doc_dict["meta"] = doc_dict.pop("metadata")

    # Check if document already exists
    existing_doc = get_document_by_filename(db, doc.filename)

    if existing_doc:
        # Delete all existing nodes (cascade will handle related records)
        db.execute(
            select(DocumentNode).where(DocumentNode.document_id == existing_doc.id)
        ).scalars().all()
        for node in db.execute(
            select(DocumentNode).where(DocumentNode.document_id == existing_doc.id)
        ).scalars():
            db.delete(node)

        # Update existing document
        for key, value in doc_dict.items():
            setattr(existing_doc, key, value)
        db.commit()
        db.refresh(existing_doc)
        return existing_doc

    # Create new document
    db_doc = Document(**doc_dict)
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    return db_doc


def get_document_by_filename(db: Session, filename: str) -> Document | None:
    """
    Retrieve a document by its filename.

    Args:
        db: Database session
        filename: Name of the file

    Returns:
        Document instance if found, None otherwise
    """
    return db.execute(
        select(Document).where(Document.filename == filename)
    ).scalar_one_or_none()


def create_document_node(db: Session, node: DocumentNodeCreate) -> DocumentNode:
    """
    Create a new document node in the database.

    Args:
        db: Database session
        node: Node data to create

    Returns:
        Created DocumentNode instance
    """
    node_dict = node.model_dump()
    # Map 'metadata' from schema to 'meta' in database model
    if "metadata" in node_dict:
        node_dict["meta"] = node_dict.pop("metadata")
    db_node = DocumentNode(**node_dict)
    db.add(db_node)
    db.commit()
    db.refresh(db_node)
    return db_node


def create_embedding(db: Session, embedding: EmbeddingCreate) -> Embedding:
    """
    Create a new embedding in the database.

    Args:
        db: Database session
        embedding: Embedding data to create

    Returns:
        Created Embedding instance
    """
    db_embedding = Embedding(**embedding.model_dump())
    db.add(db_embedding)
    db.commit()
    db.refresh(db_embedding)
    return db_embedding


def create_bm25_index(db: Session, bm25: BM25IndexCreate) -> BM25Index:
    """
    Create a new BM25 index entry in the database.

    Args:
        db: Database session
        bm25: BM25 index data to create

    Returns:
        Created BM25Index instance
    """
    db_bm25 = BM25Index(**bm25.model_dump())
    db.add(db_bm25)
    db.commit()
    db.refresh(db_bm25)
    return db_bm25


def create_contextual_chunk(
    db: Session, chunk: ContextualChunkCreate
) -> ContextualChunk:
    """
    Create a new contextual chunk in the database.

    Args:
        db: Database session
        chunk: Contextual chunk data to create

    Returns:
        Created ContextualChunk instance
    """
    db_chunk = ContextualChunk(**chunk.model_dump())
    db.add(db_chunk)
    db.commit()
    db.refresh(db_chunk)
    return db_chunk


def get_node_with_ancestors(db: Session, node_id: int) -> DocumentNode | None:
    """
    Retrieve a node with all its ancestor nodes loaded.

    Args:
        db: Database session
        node_id: ID of the node to retrieve

    Returns:
        DocumentNode with ancestors loaded, or None if not found
    """
    stmt = (
        select(DocumentNode)
        .where(DocumentNode.id == node_id)
        .options(joinedload(DocumentNode.parent))
    )
    return db.execute(stmt).scalar_one_or_none()


def get_node_ancestors(db: Session, node_id: int) -> list[DocumentNode]:
    """
    Retrieve all ancestor nodes for a given node by walking up the tree.

    Args:
        db: Database session
        node_id: ID of the starting node

    Returns:
        List of ancestor DocumentNode instances, ordered from immediate parent to root
    """
    ancestors = []
    current_node = get_node_with_ancestors(db, node_id)

    while current_node and current_node.parent_id is not None:
        parent = get_node_with_ancestors(db, current_node.parent_id)
        if parent:
            ancestors.append(parent)
            current_node = parent
        else:
            break

    return ancestors


def get_node_children(db: Session, node_id: int) -> list[DocumentNode]:
    """
    Retrieve all direct children of a node.

    Args:
        db: Database session
        node_id: ID of the parent node

    Returns:
        List of child DocumentNode instances
    """
    stmt = select(DocumentNode).where(DocumentNode.parent_id == node_id)
    return list(db.execute(stmt).scalars().all())


def get_all_leaf_nodes(
    db: Session, document_id: int | None = None
) -> list[DocumentNode]:
    """
    Retrieve all leaf nodes, optionally filtered by document.

    Args:
        db: Database session
        document_id: Optional document ID to filter by

    Returns:
        List of leaf DocumentNode instances
    """
    stmt = select(DocumentNode).where(DocumentNode.is_leaf.is_(True))
    if document_id is not None:
        stmt = stmt.where(DocumentNode.document_id == document_id)
    return list(db.execute(stmt).scalars().all())


def get_all_documents(db: Session) -> list[Document]:
    """
    Retrieve all documents from the database.

    Args:
        db: Database session

    Returns:
        List of all Document instances
    """
    return list(db.execute(select(Document)).scalars().all())
