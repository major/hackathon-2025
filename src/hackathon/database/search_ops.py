"""Database search operations (PostgreSQL FTS and BM25 index management)."""

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from hackathon.models.database import DocumentNode, MultiFieldBM25Index
from hackathon.models.schemas import MultiFieldBM25IndexCreate


def create_multifield_bm25_index(
    db: Session, index: MultiFieldBM25IndexCreate
) -> MultiFieldBM25Index:
    """
    Create a new multi-field BM25 index entry in the database.

    Args:
        db: Database session
        index: Multi-field BM25 index data to create

    Returns:
        Created MultiFieldBM25Index instance
    """
    db_index = MultiFieldBM25Index(**index.model_dump())
    db.add(db_index)
    db.commit()
    db.refresh(db_index)
    return db_index


def get_all_multifield_bm25_indexes(db: Session) -> list[MultiFieldBM25Index]:
    """
    Retrieve all multi-field BM25 index entries.

    Args:
        db: Database session

    Returns:
        List of all MultiFieldBM25Index instances
    """
    return list(db.execute(select(MultiFieldBM25Index)).scalars().all())


def search_postgres_fts(
    db: Session, query: str, top_k: int = 100
) -> list[tuple[int, float]]:
    """
    Search using PostgreSQL full-text search with ts_rank scoring.

    Args:
        db: Database session
        query: Search query (supports phrases with "quotes", prefix with word:*)
        top_k: Maximum number of results to return

    Returns:
        List of (node_id, score) tuples ordered by score descending
    """
    # Convert query to tsquery format
    # plainto_tsquery handles basic queries, to_tsquery handles advanced syntax
    # Using plainto_tsquery for simplicity (handles special chars safely)
    query_func = func.plainto_tsquery("english", query)

    # Search with ts_rank for scoring
    stmt = (
        select(
            DocumentNode.id,
            func.ts_rank(DocumentNode.text_search, query_func).label("score"),
        )
        .where(DocumentNode.text_search.op("@@")(query_func))  # Match operator
        .where(DocumentNode.is_leaf.is_(True))
        .order_by(text("score DESC"))
        .limit(top_k)
    )

    results = db.execute(stmt).all()
    return [(row.id, float(row.score)) for row in results]
