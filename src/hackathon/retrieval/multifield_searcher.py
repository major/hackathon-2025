"""Multi-field BM25 search with five-way Reciprocal Rank Fusion (4 BM25 + PostgreSQL FTS)."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from hackathon.models.database import DocumentNode
from hackathon.models.schemas import SearchResult
from hackathon.processing.bm25_search import (
    load_multifield_bm25_indexes,
    search_multifield,
)
from hackathon.utils.logging import get_logger

logger = get_logger(__name__)


class MultiFieldBM25Searcher:
    """
    Multi-field search using five-way Reciprocal Rank Fusion.

    Combines five ranking signals:
    1. BM25 full text search
    2. BM25 headings search (hierarchical context)
    3. BM25 summaries search (first 1-2 sentences)
    4. BM25 contextual text search (LLM-generated contextual summaries)
    5. PostgreSQL full-text search (exact matching with tsvector)
    """

    def __init__(self, db: Session) -> None:
        """
        Initialize the multi-field BM25 searcher.

        Args:
            db: Database session
        """
        self.db = db

        # Load persistent bm25 indexes
        try:
            (
                self.full_text_retriever,
                self.headings_retriever,
                self.summaries_retriever,
                self.contextual_text_retriever,
                self.node_ids,
            ) = load_multifield_bm25_indexes()
            logger.info(
                "Loaded multi-field indexes (BM25: full_text, headings, summaries, contextual_text + PostgreSQL FTS)"
            )
        except FileNotFoundError as e:
            logger.warning(f"Multi-field BM25 indexes not found: {e}")
            self.full_text_retriever = None
            self.headings_retriever = None
            self.summaries_retriever = None
            self.contextual_text_retriever = None
            self.node_ids = []

    def _calculate_retrieval_k(
        self, top_k: int, use_reranker: bool, rerank_candidates: int
    ) -> int:
        """
        Calculate how many candidates to retrieve based on reranking settings.

        Args:
            top_k: Number of final results needed
            use_reranker: Whether reranking will be applied
            rerank_candidates: Number of candidates for reranking

        Returns:
            Number of candidates to retrieve (capped to corpus size)
        """
        retrieval_k = rerank_candidates if use_reranker else top_k
        corpus_size = len(self.node_ids)

        if retrieval_k > corpus_size:
            logger.warning(
                f"Requested {retrieval_k} candidates but corpus only has {corpus_size} documents. "
                f"Using corpus_size={corpus_size} instead."
            )
            retrieval_k = corpus_size

        return retrieval_k

    def _fetch_search_results(
        self, results: list[tuple[int, float]]
    ) -> list[SearchResult]:
        """
        Convert node IDs and scores to SearchResult objects.

        Args:
            results: List of (node_id, score) tuples

        Returns:
            List of SearchResult objects with node details
        """
        search_results = []
        for node_id, rrf_score in results:
            stmt = select(DocumentNode).where(DocumentNode.id == node_id)
            node = self.db.execute(stmt).scalar_one_or_none()

            if node:
                result = SearchResult(
                    node_id=node.id,
                    document_id=node.document_id,
                    text_content=node.text_content or "",
                    node_type=node.node_type,
                    node_path=node.node_path,
                    score=rrf_score,
                    metadata=node.meta,
                )
                search_results.append(result)

        return search_results

    def _apply_reranking(
        self, query: str, search_results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """
        Apply semantic reranking to search results.

        Args:
            query: Search query
            search_results: Initial search results
            top_k: Number of top results to return

        Returns:
            Reranked search results
        """
        logger.info(
            f"Applying IBM Watsonx reranking to {len(search_results)} candidates"
        )
        from hackathon.retrieval.reranker import rerank_results

        return rerank_results(query=query, results=search_results, top_n=top_k)

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_reranker: bool = False,
        rerank_candidates: int = 50,
    ) -> list[SearchResult]:
        """
        Perform five-way search with RRF, with optional semantic reranking.

        Combines BM25 (full_text, headings, summaries, contextual_text) with PostgreSQL FTS
        using Reciprocal Rank Fusion for improved retrieval accuracy.

        Optionally applies IBM Watsonx reranking for semantic relevance.

        Args:
            query: Search query
            top_k: Number of top results to return
            use_reranker: Whether to use IBM Watsonx semantic reranking (default: False)
            rerank_candidates: Number of candidates to retrieve before reranking (default: 50)
                Only used if use_reranker=True. Recommended: 50-100 for best quality.

        Returns:
            List of SearchResult objects
        """
        if not self.full_text_retriever or not self.node_ids:
            logger.warning(
                "Multi-field BM25 indexes not loaded, returning empty results"
            )
            return []

        # Type narrowing: after the check above, we know all retrievers are loaded
        assert self.headings_retriever is not None
        assert self.summaries_retriever is not None
        assert self.contextual_text_retriever is not None

        # Calculate how many candidates to retrieve
        retrieval_k = self._calculate_retrieval_k(
            top_k, use_reranker, rerank_candidates
        )

        # Search across all five fields using RRF (4 BM25 + PostgreSQL FTS)
        results = search_multifield(
            query=query,
            full_text_retriever=self.full_text_retriever,
            headings_retriever=self.headings_retriever,
            summaries_retriever=self.summaries_retriever,
            contextual_text_retriever=self.contextual_text_retriever,
            node_ids=self.node_ids,
            db=self.db,
            top_k=retrieval_k,
        )

        # Fetch node details for results
        search_results = self._fetch_search_results(results)

        # Apply reranking if requested
        if use_reranker:
            search_results = self._apply_reranking(query, search_results, top_k)

        return search_results
