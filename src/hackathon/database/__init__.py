"""Database connection and operations."""

from hackathon.database.connection import get_db, init_db, reset_db, session_scope
from hackathon.database.operations import (
    create_bm25_index,
    create_contextual_chunk,
    create_document,
    create_document_node,
    create_embedding,
    get_all_documents,
    get_all_leaf_nodes,
    get_document_by_filename,
    get_node_ancestors,
    get_node_children,
    get_node_with_ancestors,
)

__all__ = [
    "create_bm25_index",
    "create_contextual_chunk",
    "create_document",
    "create_document_node",
    "create_embedding",
    "get_all_documents",
    "get_all_leaf_nodes",
    "get_db",
    "get_document_by_filename",
    "get_node_ancestors",
    "get_node_children",
    "get_node_with_ancestors",
    "init_db",
    "reset_db",
    "session_scope",
]
