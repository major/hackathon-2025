"""IBM Watsonx reranker for improved semantic relevance."""

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Rerank

from hackathon.config import get_settings
from hackathon.models.schemas import SearchResult
from hackathon.utils.logging import get_logger

logger = get_logger(__name__)

# Model token limit constants
MAX_DOCUMENT_CHARS = 800  # Conservative ~200 tokens to fit within 512 token limit


def _truncate_documents(documents: list[str]) -> tuple[list[str], int]:
    """
    Truncate documents that exceed the model's token limit.

    Args:
        documents: List of document strings

    Returns:
        Tuple of (truncated_documents, truncated_count)
    """
    truncated_documents = []
    truncated_count = 0

    for doc in documents:
        if len(doc) > MAX_DOCUMENT_CHARS:
            truncated_documents.append(doc[:MAX_DOCUMENT_CHARS] + "...")
            truncated_count += 1
        else:
            truncated_documents.append(doc)

    return truncated_documents, truncated_count


def _validate_query_length(query: str) -> None:
    """
    Validate query length and warn if too short for semantic reranking.

    Args:
        query: User's search query
    """
    query_words = query.strip().split()
    if len(query_words) <= 2:
        logger.warning(
            f"Query is very short ({len(query_words)} word{'s' if len(query_words) > 1 else ''}): '{query}'. "
            "Semantic reranking may reduce precision for keyword searches. "
            "Consider using BM25-only (without --rerank) for exact keyword matches."
        )


def _map_reranked_results(
    reranked_items: list[dict],
    original_results: list[SearchResult],
) -> list[SearchResult]:
    """
    Map reranked items back to SearchResult objects with updated scores.

    Args:
        reranked_items: List of reranked items from Watsonx API
        original_results: Original SearchResult objects

    Returns:
        List of SearchResult objects with updated scores
    """
    reranked_results = []
    for rank, item in enumerate(reranked_items, start=1):
        original_idx = item.get("index", 0)
        rerank_score = item.get("score", 0.0)

        logger.debug(
            f"Rerank position {rank}: original_idx={original_idx}, "
            f"score={rerank_score:.4f}"
        )

        # Get the original SearchResult
        if 0 <= original_idx < len(original_results):
            original_result = original_results[original_idx]

            logger.debug(
                f"  -> Mapped to: BM25 rank={original_idx + 1}, "
                f"text preview={original_result.text_content[:80]}..."
            )

            # Update metadata with reranking info
            if original_result.metadata is None:
                original_result.metadata = {}
            original_result.metadata["original_score"] = original_result.score
            original_result.metadata["bm25_rank"] = original_idx + 1
            original_result.metadata["rerank_score"] = rerank_score

            # Update the score to the rerank score
            original_result.score = rerank_score

            reranked_results.append(original_result)

    return reranked_results


class WatsonxReranker:
    """
    Rerank search results using IBM Watsonx reranker models.

    Uses cross-encoder models for semantic relevance scoring, which
    significantly improves precision over BM25/keyword-only approaches.

    Uses the native IBM Watsonx AI SDK (no LangChain/LlamaIndex/Haystack).
    """

    def __init__(
        self,
        api_key: str | None = None,
        project_id: str | None = None,
        model: str | None = None,
    ):
        """
        Initialize Watsonx reranker.

        Args:
            api_key: Watsonx API key (defaults to config)
            project_id: Watsonx project ID (defaults to config)
            model: Watsonx rerank model to use (defaults to config)
                - "cross-encoder/ms-marco-minilm-l-12-v2" (default, MS MARCO trained)
        """
        settings = get_settings()
        self.api_key = api_key or settings.watsonx_api_key
        self.project_id = project_id or settings.watsonx_project_id
        self.model = model or settings.watsonx_reranker_model

        if not self.api_key:
            msg = (
                "Watsonx API key not found. Please set WATSONX_API_KEY in .env or pass api_key. "
                "Get your key at: https://cloud.ibm.com/iam/apikeys"
            )
            raise ValueError(msg)

        if not self.project_id:
            msg = "Watsonx project ID not found. Please set WATSONX_PROJECT_ID in .env or pass project_id"
            raise ValueError(msg)

        # Initialize Watsonx credentials
        self.credentials = Credentials(
            url=settings.watsonx_url,
            api_key=self.api_key,
        )

        # Initialize reranker using native SDK (no LangChain!)
        self.reranker = Rerank(
            model_id=self.model,
            credentials=self.credentials,
            project_id=self.project_id,
        )

        logger.info(f"Initialized Watsonx reranker with model: {self.model}")

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int = 5,
    ) -> list[SearchResult]:
        """
        Rerank search results using Watsonx's semantic reranking.

        Args:
            query: User's search query
            results: List of SearchResult objects from initial retrieval
            top_n: Number of top results to return after reranking

        Returns:
            Reranked list of SearchResult objects (top_n items)
        """
        if not results:
            logger.warning("No results to rerank")
            return []

        if len(results) <= top_n:
            logger.info(
                f"Only {len(results)} results, less than top_n={top_n}, reranking anyway"
            )

        # Extract and prepare documents for reranking
        documents: list[str] = [r.text_content for r in results]
        documents, truncated_count = _truncate_documents(documents)

        if truncated_count > 0:
            logger.info(
                f"Truncated {truncated_count}/{len(documents)} documents to fit 512 token limit"
            )

        # Validate query length and warn if needed
        _validate_query_length(query)

        logger.info(
            f"Reranking {len(documents)} documents with Watsonx ({self.model})..."
        )

        # Call Watsonx rerank API using native SDK
        try:
            response = self.reranker.generate(
                query=query,
                inputs=documents,  # pyright: ignore[reportArgumentType]
            )

            logger.debug(f"Watsonx rerank response: {response}")

            # Parse and validate response
            reranked_items = response.get("results", [])
            if not reranked_items:
                logger.warning(
                    "Watsonx returned no results in 'results' key. Full response: %s",
                    response,
                )
                logger.warning("Falling back to original BM25 ranking")
                return results[:top_n]

            # Sort by score (descending) and take top_n
            reranked_items = sorted(
                reranked_items, key=lambda x: x.get("score", 0), reverse=True
            )[:top_n]

            # Map reranked items back to SearchResults
            reranked_results = _map_reranked_results(reranked_items, results)

            logger.info(f"Reranking complete: returned {len(reranked_results)} results")
            return reranked_results

        except Exception as e:
            logger.error(f"Watsonx reranking failed: {e}")
            logger.warning("Falling back to original BM25 ranking")
            return results[:top_n]


def rerank_results(
    query: str,
    results: list[SearchResult],
    top_n: int = 5,
    model: str | None = None,
) -> list[SearchResult]:
    """
    Convenience function to rerank results without creating a reranker instance.

    Args:
        query: User's search query
        results: List of SearchResult objects from initial retrieval
        top_n: Number of top results to return after reranking
        model: Watsonx rerank model to use (defaults to config)

    Returns:
        Reranked list of SearchResult objects (top_n items)
    """
    reranker = WatsonxReranker(model=model)
    return reranker.rerank(query, results, top_n)
