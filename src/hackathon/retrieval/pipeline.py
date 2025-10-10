"""End-to-end retrieval pipeline orchestration.

This module provides a high-level interface for search operations,
encapsulating the searcher, expander, and reranker into a single pipeline.
"""

from sqlalchemy.orm import Session

from hackathon.models.schemas import SearchResult
from hackathon.retrieval.context_expansion import ContextExpander
from hackathon.retrieval.multifield_searcher import MultiFieldBM25Searcher
from hackathon.utils.logging import get_logger

logger = get_logger(__name__)


class RetrievalPipeline:
    """
    End-to-end retrieval pipeline orchestration.

    Combines BM25 search, optional reranking, and context expansion
    into a single easy-to-use interface.

    Example:
        >>> from hackathon.database import get_db
        >>> db = next(get_db())
        >>> pipeline = RetrievalPipeline(db, use_reranker=True)
        >>> results = pipeline.search("how to configure logging", top_k=5)
    """

    def __init__(self, db: Session, use_reranker: bool = False):
        """
        Initialize the retrieval pipeline.

        Args:
            db: Database session
            use_reranker: Whether to use IBM Watsonx semantic reranking
        """
        self.db = db
        self.use_reranker = use_reranker

        # Initialize components
        self.searcher = MultiFieldBM25Searcher(db)
        self.expander = ContextExpander(db)

        logger.info(
            f"Initialized retrieval pipeline (reranker: {'enabled' if use_reranker else 'disabled'})"
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        rerank_candidates: int = 50,
        expand_context: bool = False,
        neighbors_before: int = 0,
        neighbors_after: int = 0,
    ) -> list[SearchResult]:
        """
        Perform end-to-end search with optional expansion.

        Args:
            query: Search query
            top_k: Number of top results to return
            rerank_candidates: Number of candidates for reranking (if enabled)
            expand_context: Whether to expand context for results
            neighbors_before: Number of neighbors before each result
            neighbors_after: Number of neighbors after each result

        Returns:
            List of SearchResult objects with optional context expansion
        """
        # Perform search (with optional reranking)
        # Note: return_expanded_queries=False ensures we get list[SearchResult], not a tuple
        search_results = self.searcher.search(
            query=query,
            top_k=top_k,
            use_reranker=self.use_reranker,
            rerank_candidates=rerank_candidates,
            return_expanded_queries=False,
        )

        # Type narrowing: when return_expanded_queries=False, we get list[SearchResult]
        assert isinstance(search_results, list), "Expected list[SearchResult]"
        results = search_results

        # Expand context if requested
        if expand_context and results:
            results = self._expand_results(results, neighbors_before, neighbors_after)

        return results

    def _expand_results(
        self,
        results: list[SearchResult],
        neighbors_before: int,
        neighbors_after: int,
    ) -> list[SearchResult]:
        """
        Expand context for search results.

        Args:
            results: List of SearchResult objects
            neighbors_before: Number of neighbors before each result
            neighbors_after: Number of neighbors after each result

        Returns:
            List of SearchResult objects with expanded context
        """
        from sqlalchemy import select

        from hackathon.models.database import DocumentNode

        expanded_results = []
        for result in results:
            # Get the node
            stmt = select(DocumentNode).where(DocumentNode.id == result.node_id)
            node = self.db.execute(stmt).scalar_one_or_none()

            if node:
                # Get expanded context if neighbors requested
                if neighbors_before > 0 or neighbors_after > 0:
                    context = self.expander.get_neighbor_context(
                        node, before=neighbors_before, after=neighbors_after
                    )
                    result.context = context

                expanded_results.append(result)

        return expanded_results

    def get_full_document(self, result: SearchResult) -> str:
        """
        Get the full document context for a search result.

        Args:
            result: SearchResult object

        Returns:
            Full document text
        """
        from sqlalchemy import select

        from hackathon.models.database import DocumentNode

        stmt = select(DocumentNode).where(DocumentNode.id == result.node_id)
        node = self.db.execute(stmt).scalar_one_or_none()

        if not node:
            logger.warning(f"Node {result.node_id} not found")
            return ""

        return self.expander.get_full_document_context(node)
