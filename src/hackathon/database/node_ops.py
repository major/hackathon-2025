"""Document node CRUD operations."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from hackathon.models.database import DocumentNode
from hackathon.models.schemas import DocumentNodeCreate


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


def get_neighbors_before(
    db: Session, node: DocumentNode, count: int = 1
) -> list[DocumentNode]:
    """
    Get N neighboring chunks before the given node in document order.

    Args:
        db: Database session
        node: The reference node
        count: Number of neighbors to retrieve

    Returns:
        List of DocumentNode instances ordered from oldest to newest (furthest to closest)
    """
    if node.position is None:
        return []

    stmt = (
        select(DocumentNode)
        .where(DocumentNode.document_id == node.document_id)
        .where(DocumentNode.is_leaf.is_(True))
        .where(DocumentNode.position < node.position)
        .order_by(DocumentNode.position.desc())
        .limit(count)
    )

    # Results are in descending order, reverse to get oldest->newest
    results = list(db.execute(stmt).scalars().all())
    return list(reversed(results))


def get_neighbors_after(
    db: Session, node: DocumentNode, count: int = 1
) -> list[DocumentNode]:
    """
    Get N neighboring chunks after the given node in document order.

    Args:
        db: Database session
        node: The reference node
        count: Number of neighbors to retrieve

    Returns:
        List of DocumentNode instances ordered from closest to furthest
    """
    if node.position is None:
        return []

    stmt = (
        select(DocumentNode)
        .where(DocumentNode.document_id == node.document_id)
        .where(DocumentNode.is_leaf.is_(True))
        .where(DocumentNode.position > node.position)
        .order_by(DocumentNode.position.asc())
        .limit(count)
    )

    return list(db.execute(stmt).scalars().all())


def get_neighbors(
    db: Session, node: DocumentNode, before: int = 1, after: int = 1
) -> tuple[list[DocumentNode], list[DocumentNode]]:
    """
    Get neighboring chunks both before and after the given node.

    Args:
        db: Database session
        node: The reference node
        before: Number of neighbors to retrieve before
        after: Number of neighbors to retrieve after

    Returns:
        Tuple of (before_neighbors, after_neighbors) where each is a list of DocumentNode instances
    """
    before_neighbors = get_neighbors_before(db, node, before)
    after_neighbors = get_neighbors_after(db, node, after)
    return before_neighbors, after_neighbors
