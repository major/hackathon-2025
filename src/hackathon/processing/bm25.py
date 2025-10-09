"""BM25 indexing for keyword-based search."""

import re

from sqlalchemy.orm import Session

from hackathon.database.operations import create_bm25_index
from hackathon.models.schemas import BM25IndexCreate


def tokenize_text(text: str) -> list[str]:
    """
    Tokenize text for BM25 indexing.

    Args:
        text: Input text to tokenize

    Returns:
        List of tokens (lowercase, alphanumeric)
    """
    # Convert to lowercase
    text = text.lower()

    # Split on non-alphanumeric characters
    tokens = re.findall(r"\b\w+\b", text)

    # Filter out very short tokens (single characters)
    return [t for t in tokens if len(t) > 1]


def create_bm25_index_for_node(db: Session, node_id: int, text: str) -> None:
    """
    Create BM25 index entry for a document node.

    Args:
        db: Database session
        node_id: ID of the document node
        text: Text content to index
    """
    tokens = tokenize_text(text)

    bm25_data = BM25IndexCreate(node_id=node_id, tokens=tokens, token_count=len(tokens))

    create_bm25_index(db, bm25_data)


def batch_create_bm25_indices(
    db: Session, node_texts: list[tuple[int, str]], batch_size: int = 100
) -> None:
    """
    Create BM25 indices for multiple nodes in batches.

    Args:
        db: Database session
        node_texts: List of (node_id, text) tuples
        batch_size: Number of indices to create per batch
    """
    for i in range(0, len(node_texts), batch_size):
        batch = node_texts[i : i + batch_size]
        for node_id, text in batch:
            if text:  # Only create index if text is not empty
                create_bm25_index_for_node(db, node_id, text)

        # Commit after each batch
        db.commit()
