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

    def _expand_query(self, query: str, num_variations: int) -> list[str]:
        """
        Expand query into semantic variations using IBM Watsonx.

        Args:
            query: Original search query
            num_variations: Number of variations to generate

        Returns:
            List of query variations (original + generated)
        """
        logger.info(f"Expanding query '{query}' into {num_variations} variations")
        from hackathon.retrieval.query_expansion import QueryExpander

        expander = QueryExpander(num_variations=num_variations)
        variations = expander.expand_query(query)
        logger.info(f"Query expanded to {len(variations)} variations: {variations}")
        return variations

    def _search_with_expanded_queries(
        self, queries: list[str], top_k: int
    ) -> list[tuple[int, float]]:
        """
        Search with multiple query variations and merge results using RRF.

        Args:
            queries: List of query variations to search with
            top_k: Number of top results to return

        Returns:
            List of (node_id, rrf_score) tuples
        """
        from hackathon.processing.bm25_search import reciprocal_rank_fusion

        logger.info(f"Searching with {len(queries)} query variations")

        # Search with each query variation
        all_rankings = []
        for i, q in enumerate(queries, 1):
            logger.info(f"Searching with variation {i}/{len(queries)}: '{q}'")
            results = search_multifield(
                query=q,
                full_text_retriever=self.full_text_retriever,  # type: ignore
                headings_retriever=self.headings_retriever,  # type: ignore
                summaries_retriever=self.summaries_retriever,  # type: ignore
                contextual_text_retriever=self.contextual_text_retriever,  # type: ignore
                node_ids=self.node_ids,
                db=self.db,
                top_k=top_k * 2,  # Get more candidates per query for better RRF
            )

            # Convert results to ranking (list of node_ids in rank order)
            ranking = [node_id for node_id, _ in results]
            all_rankings.append(ranking)

        # Merge all rankings using RRF
        # Create a mapping from node_id to index for RRF
        all_node_ids = list(
            set(node_id for ranking in all_rankings for node_id in ranking)
        )
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(all_node_ids)}

        # Convert node_id rankings to index rankings
        index_rankings = [
            [node_id_to_idx[node_id] for node_id in ranking] for ranking in all_rankings
        ]

        # Apply RRF
        rrf_results = reciprocal_rank_fusion(index_rankings)

        # Convert back to node_ids and take top_k
        results_with_node_ids = [
            (all_node_ids[idx], score) for idx, score in rrf_results[:top_k]
        ]

        logger.info(
            f"Merged {len(queries)} query variations into {len(results_with_node_ids)} results"
        )
        return results_with_node_ids

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
        use_query_expansion: bool = False,
        num_query_variations: int = 3,
        return_expanded_queries: bool = False,
    ) -> list[SearchResult] | tuple[list[SearchResult], list[str]]:
        """
        Perform five-way search with RRF, with optional query expansion and semantic reranking.

        Combines BM25 (full_text, headings, summaries, contextual_text) with PostgreSQL FTS
        using Reciprocal Rank Fusion for improved retrieval accuracy.

        Optionally expands the query into semantic variations for better recall.
        Optionally applies IBM Watsonx reranking for semantic relevance.

        Args:
            query: Search query
            top_k: Number of top results to return
            use_reranker: Whether to use IBM Watsonx semantic reranking (default: False)
            rerank_candidates: Number of candidates to retrieve before reranking (default: 50)
                Only used if use_reranker=True. Recommended: 50-100 for best quality.
            use_query_expansion: Whether to expand query with semantic variations (default: False)
            num_query_variations: Number of query variations to generate (default: 3)
                Only used if use_query_expansion=True.
            return_expanded_queries: Whether to return the expanded queries (default: False)
                If True, returns tuple of (results, expanded_queries)

        Returns:
            List of SearchResult objects, or tuple of (results, expanded_queries) if return_expanded_queries=True
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

        # Expand query if requested
        queries_to_search = [query]
        if use_query_expansion:
            queries_to_search = self._expand_query(query, num_query_variations)

        # Search with all query variations and merge results
        if len(queries_to_search) == 1:
            # Single query - use standard search
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
        else:
            # Multiple queries - search with each and merge via RRF
            results = self._search_with_expanded_queries(queries_to_search, retrieval_k)

        # Fetch node details for results
        search_results = self._fetch_search_results(results)

        # Apply reranking if requested (use original query, not variations)
        if use_reranker:
            search_results = self._apply_reranking(query, search_results, top_k)

        # Return results with expanded queries if requested
        if return_expanded_queries:
            return search_results, queries_to_search
        return search_results
