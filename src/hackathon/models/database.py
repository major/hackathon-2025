"""SQLAlchemy database models for the RAG system."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator


class TSVectorType(TypeDecorator):
    """
    Cross-database TSVECTOR type.

    Uses PostgreSQL's TSVECTOR for production, falls back to TEXT for SQLite tests.
    """

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        """Load the appropriate type for the database dialect."""
        if dialect.name == "postgresql":
            return dialect.type_descriptor(TSVECTOR())
        else:
            # For SQLite and other databases, use TEXT
            return dialect.type_descriptor(Text())


class Base(DeclarativeBase):
    """Base class for all database models."""


class Document(Base):
    """Store PDF document metadata."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    filepath: Mapped[str] = mapped_column(String(512), nullable=False)
    processing_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    meta: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)

    # Relationships
    nodes: Mapped[list["DocumentNode"]] = relationship(
        "DocumentNode", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentNode(Base):
    """
    Store document chunks with flat structure.

    Uses sequential positions for neighbor lookup and PostgreSQL FTS for exact matching.
    """

    __tablename__ = "document_nodes"
    __table_args__ = (
        UniqueConstraint("document_id", "node_path", name="uq_document_node_path"),
        # Composite index for neighbor queries: WHERE document_id = X AND is_leaf = Y AND position < Z
        Index("idx_neighbor_lookup", "document_id", "is_leaf", "position"),
        # PostgreSQL-specific indexes (ignored by other databases)
        # GIN index for full-text search
        Index("idx_text_search", "text_search", postgresql_using="gin"),
        # Partial GIN index for FTS on leaf nodes only (reduces index size)
        Index(
            "idx_text_search_leaf",
            "text_search",
            postgresql_using="gin",
            postgresql_where=text("is_leaf = true"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    node_type: Mapped[str] = mapped_column(String(50), nullable=False)
    text_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_leaf: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    node_path: Mapped[str] = mapped_column(
        String(512), nullable=False
    )  # Simple sequential path like "chunk_0", "chunk_1"
    position: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )  # Sequential position in document (0, 1, 2...) for neighbor lookup
    meta: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)
    text_search: Mapped[str | None] = mapped_column(
        TSVectorType, nullable=True
    )  # PostgreSQL full-text search vector (TEXT in SQLite)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="nodes")
    bm25_index: Mapped["MultiFieldBM25Index | None"] = relationship(
        "MultiFieldBM25Index",
        back_populates="node",
        uselist=False,
        cascade="all, delete-orphan",
    )


class MultiFieldBM25Index(Base):
    """
    Store multi-field BM25 index data for each document node.

    This implements contextual retrieval with four searchable fields:
    - full_text: Complete chunk text (primary content)
    - headings: Hierarchical heading context for conceptual matching
    - summary: First 1-2 sentences for topic matching
    - contextual_text: LLM-generated contextual summary + chunk text (Anthropic pattern)
    """

    __tablename__ = "multifield_bm25_index"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("document_nodes.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    full_text: Mapped[str] = mapped_column(Text, nullable=False)
    headings: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # e.g., "Config > Logging > Settings"
    summary: Mapped[str] = mapped_column(Text, nullable=False)  # First 1-2 sentences
    contextual_text: Mapped[str] = mapped_column(
        Text, nullable=False, default=""
    )  # Contextual summary + original text
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    # Relationships
    node: Mapped["DocumentNode"] = relationship(
        "DocumentNode", back_populates="bm25_index"
    )
