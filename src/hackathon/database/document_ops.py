"""Document CRUD operations."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from hackathon.models.database import Document, DocumentNode
from hackathon.models.schemas import DocumentCreate


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


def get_document_by_id(db: Session, document_id: int) -> Document | None:
    """
    Retrieve a document by its ID.

    Args:
        db: Database session
        document_id: Document ID

    Returns:
        Document instance if found, None otherwise
    """
    return db.execute(
        select(Document).where(Document.id == document_id)
    ).scalar_one_or_none()


def get_all_documents(db: Session) -> list[Document]:
    """
    Retrieve all documents from the database.

    Args:
        db: Database session

    Returns:
        List of all Document instances
    """
    return list(db.execute(select(Document)).scalars().all())
