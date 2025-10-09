"""Hybrid search combining BM25 and semantic search."""

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from hackathon.models.database import BM25Index, DocumentNode, Embedding
from hackathon.models.schemas import SearchResult
from hackathon.processing.bm25 import tokenize_text
from hackathon.processing.embedder import EmbeddingGenerator


class HybridSearcher:
    """Hybrid search using BM25 and semantic similarity."""

    def __init__(self, db: Session) -> None:
        """
        Initialize the hybrid searcher.

        Args:
            db: Database session
        """
        self.db = db
        self.embedding_generator = EmbeddingGenerator()

    def bm25_search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of (node_id, score) tuples
        """
        # Get all BM25 indices
        stmt = select(BM25Index).options(joinedload(BM25Index.node))
        bm25_indices = list(self.db.execute(stmt).scalars().all())

        if not bm25_indices:
            return []

        # Build BM25 index
        corpus = [idx.tokens for idx in bm25_indices]
        bm25 = BM25Okapi(corpus)

        # Tokenize query
        query_tokens = tokenize_text(query)

        # Get scores
        scores = bm25.get_scores(query_tokens)

        # Get top k
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (bm25_indices[i].node_id, float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    def semantic_search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Perform semantic similarity search using embeddings.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of (node_id, score) tuples
        """
        # Generate query embedding
        query_vector = self.embedding_generator.embed_text(query)

        # Query embeddings with vector similarity
        # Using pgvector's cosine distance operator (<=>)
        stmt = (
            select(
                Embedding.node_id,
                Embedding.vector.cosine_distance(query_vector).label("distance"),
            )
            .order_by("distance")
            .limit(top_k)
        )

        results = self.db.execute(stmt).all()

        # Convert distance to similarity score (1 - distance for cosine)
        return [(node_id, 1 - float(distance)) for node_id, distance in results]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        bm25_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ) -> list[SearchResult]:
        """
        Perform hybrid search combining BM25 and semantic search.

        Args:
            query: Search query
            top_k: Number of top results to return
            bm25_weight: Weight for BM25 scores (0-1)
            semantic_weight: Weight for semantic scores (0-1)

        Returns:
            List of SearchResult objects
        """
        # Get BM25 results
        bm25_results = dict(self.bm25_search(query, top_k=top_k * 2))

        # Get semantic results
        semantic_results = dict(self.semantic_search(query, top_k=top_k * 2))

        # Combine scores
        all_node_ids = set(bm25_results.keys()) | set(semantic_results.keys())
        combined_scores = {}

        # Normalize scores to 0-1 range
        bm25_scores = list(bm25_results.values())
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0

        for node_id in all_node_ids:
            bm25_score = (
                bm25_results.get(node_id, 0.0) / max_bm25 if max_bm25 > 0 else 0.0
            )
            semantic_score = semantic_results.get(node_id, 0.0)

            combined_score = (bm25_weight * bm25_score) + (
                semantic_weight * semantic_score
            )
            combined_scores[node_id] = combined_score

        # Sort by combined score
        sorted_node_ids = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        # Fetch node details
        results = []
        for node_id, score in sorted_node_ids:
            stmt = select(DocumentNode).where(DocumentNode.id == node_id)
            node = self.db.execute(stmt).scalar_one_or_none()

            if node:
                result = SearchResult(
                    node_id=node.id,
                    document_id=node.document_id,
                    text_content=node.text_content or "",
                    node_type=node.node_type,
                    node_path=node.node_path,
                    score=score,
                    metadata=node.meta,
                )
                results.append(result)

        return results
