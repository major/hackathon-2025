"""SQLAlchemy database models for the RAG system."""

from datetime import UTC, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


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
    """Store parsed document tree structure with hierarchical relationships."""

    __tablename__ = "document_nodes"
    __table_args__ = (
        UniqueConstraint("document_id", "node_path", name="uq_document_node_path"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    parent_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("document_nodes.id", ondelete="CASCADE"), nullable=True
    )
    node_type: Mapped[str] = mapped_column(String(50), nullable=False)
    text_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_leaf: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    node_path: Mapped[str] = mapped_column(
        String(512), nullable=False
    )  # e.g., "0.1.2" for hierarchical path
    meta: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="nodes")
    parent: Mapped["DocumentNode | None"] = relationship(
        "DocumentNode", remote_side=[id], back_populates="children"
    )
    children: Mapped[list["DocumentNode"]] = relationship(
        "DocumentNode", back_populates="parent", cascade="all, delete-orphan"
    )
    embedding: Mapped["Embedding | None"] = relationship(
        "Embedding", back_populates="node", uselist=False, cascade="all, delete-orphan"
    )
    bm25_data: Mapped["BM25Index | None"] = relationship(
        "BM25Index", back_populates="node", uselist=False, cascade="all, delete-orphan"
    )
    contextual_chunk: Mapped["ContextualChunk | None"] = relationship(
        "ContextualChunk",
        back_populates="node",
        uselist=False,
        cascade="all, delete-orphan",
    )


class Embedding(Base):
    """Store vector embeddings for semantic search."""

    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("document_nodes.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    vector: Mapped[Vector] = mapped_column(
        Vector(384), nullable=False
    )  # granite-embedding dim
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    # Relationships
    node: Mapped["DocumentNode"] = relationship(
        "DocumentNode", back_populates="embedding"
    )


class BM25Index(Base):
    """Store BM25 tokenized data for keyword-based search."""

    __tablename__ = "bm25_index"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("document_nodes.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    tokens: Mapped[list[str]] = mapped_column(JSON, nullable=False)  # Tokenized text
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Relationships
    node: Mapped["DocumentNode"] = relationship(
        "DocumentNode", back_populates="bm25_data"
    )


class ContextualChunk(Base):
    """Store chunks with contextual information for Anthropic's Contextual Retrieval."""

    __tablename__ = "contextual_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("document_nodes.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    contextual_summary: Mapped[str] = mapped_column(Text, nullable=False)
    contextualized_text: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # Summary + original text
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    # Relationships
    node: Mapped["DocumentNode"] = relationship(
        "DocumentNode", back_populates="contextual_chunk"
    )
