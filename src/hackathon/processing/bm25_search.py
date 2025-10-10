"""Multi-field BM25 search and ranking (query-time operations)."""

from pathlib import Path

import bm25s

from hackathon.config import get_settings

# 🔇 Logging configured centrally in hackathon/__init__.py


def load_multifield_bm25_indexes() -> tuple[
    bm25s.BM25, bm25s.BM25, bm25s.BM25, bm25s.BM25, list[int]
]:
    """
    Load all four BM25 indexes.

    Returns:
        Tuple of (full_text_retriever, headings_retriever, summaries_retriever,
                 contextual_text_retriever, node_ids)
    """
    settings = get_settings()
    index_path = Path(settings.bm25_index_path)

    if not index_path.exists():
        msg = f"BM25 indexes not found at {index_path}. Run 'uv run process' first."
        raise FileNotFoundError(msg)

    # 🤫 Load retrievers (silently - no progress bars)
    full_text_retriever = bm25s.BM25.load(
        str(index_path / "full_text"), load_corpus=True, mmap=True
    )
    headings_retriever = bm25s.BM25.load(
        str(index_path / "headings"), load_corpus=True, mmap=True
    )
    summaries_retriever = bm25s.BM25.load(
        str(index_path / "summaries"), load_corpus=True, mmap=True
    )
    contextual_text_retriever = bm25s.BM25.load(
        str(index_path / "contextual_text"), load_corpus=True, mmap=True
    )

    # Load node_id mapping
    import json

    mapping_file = index_path / "node_ids.json"
    node_ids = json.loads(mapping_file.read_text())

    return (
        full_text_retriever,
        headings_retriever,
        summaries_retriever,
        contextual_text_retriever,
        node_ids,
    )


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Combine multiple rankings using Reciprocal Rank Fusion.

    RRF formula: score = sum(1 / (k + rank)) for each ranking list

    Args:
        rankings: List of ranking lists (each is a list of node indices)
        k: Constant for RRF (default 60, standard value from literature)

    Returns:
        List of (node_index, rrf_score) tuples, sorted by score descending
    """
    scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, node_idx in enumerate(ranking, start=1):
            if node_idx not in scores:
                scores[node_idx] = 0.0
            scores[node_idx] += 1.0 / (k + rank)

    # Sort by score descending
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


def search_multifield(
    query: str,
    full_text_retriever: bm25s.BM25,
    headings_retriever: bm25s.BM25,
    summaries_retriever: bm25s.BM25,
    contextual_text_retriever: bm25s.BM25,
    node_ids: list[int],
    db,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """
    Search across all five fields and combine results with RRF.

    Uses five-way Reciprocal Rank Fusion combining:
    1. BM25 full text search
    2. BM25 headings search
    3. BM25 summaries search
    4. BM25 contextual text search (LLM-generated context)
    5. PostgreSQL full-text search (tsvector)

    Args:
        query: Search query
        full_text_retriever: BM25 index for full text
        headings_retriever: BM25 index for headings
        summaries_retriever: BM25 index for summaries
        contextual_text_retriever: BM25 index for contextual text
        node_ids: List of node IDs corresponding to index positions
        db: Database session for PostgreSQL FTS
        top_k: Number of results to return

    Returns:
        List of (node_id, rrf_score) tuples
    """
    from hackathon.database.search_ops import search_postgres_fts

    # 🤫 Tokenize query once (silently)
    query_tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)

    # Search each BM25 index (get top 100 candidates from each)
    # We get more candidates than top_k to allow RRF to rerank effectively
    # Cap at corpus size to avoid BM25 errors
    corpus_size = len(node_ids)
    candidate_k = min(max(100, top_k * 3), corpus_size)

    # 🤫 Suppress progress bars for cleaner output
    full_text_results, _ = full_text_retriever.retrieve(
        query_tokens, k=candidate_k, show_progress=False
    )
    headings_results, _ = headings_retriever.retrieve(
        query_tokens, k=candidate_k, show_progress=False
    )
    summaries_results, _ = summaries_retriever.retrieve(
        query_tokens, k=candidate_k, show_progress=False
    )
    contextual_text_results, _ = contextual_text_retriever.retrieve(
        query_tokens, k=candidate_k, show_progress=False
    )

    # Convert BM25 results to rankings (list of doc indices in rank order)
    # bm25s returns list of dicts with 'id' keys when corpus is loaded
    full_text_ranking = [doc["id"] for doc in full_text_results[0]]
    headings_ranking = [doc["id"] for doc in headings_results[0]]
    summaries_ranking = [doc["id"] for doc in summaries_results[0]]
    contextual_text_ranking = [doc["id"] for doc in contextual_text_results[0]]

    # Get PostgreSQL FTS results
    postgres_fts_results = search_postgres_fts(db, query, top_k=candidate_k)

    # Convert PostgreSQL FTS results (node_id, score) to doc_idx ranking
    # Create a mapping from node_id to doc_idx
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    postgres_fts_ranking = [
        node_id_to_idx[node_id]
        for node_id, _ in postgres_fts_results
        if node_id in node_id_to_idx
    ]

    # Apply five-way RRF
    rrf_results = reciprocal_rank_fusion([
        full_text_ranking,
        headings_ranking,
        summaries_ranking,
        contextual_text_ranking,
        postgres_fts_ranking,
    ])

    # Convert doc indices to node IDs and take top_k
    results_with_node_ids = [
        (node_ids[doc_idx], score) for doc_idx, score in rrf_results[:top_k]
    ]

    return results_with_node_ids
