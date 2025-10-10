"""Multi-field BM25 index building (build-time operations)."""

from pathlib import Path

import bm25s

from hackathon.config import get_settings

# 🔇 Logging configured centrally in hackathon/__init__.py


def extract_summary(text: str, max_sentences: int = 2) -> str:
    """
    Extract the first N sentences from text as a summary.

    Args:
        text: Input text
        max_sentences: Number of sentences to extract (default: 2)

    Returns:
        First N sentences, or full text if fewer sentences exist
    """
    if not text:
        return ""

    # Simple sentence splitting (handles . ! ?)
    import re

    sentences = re.split(r"[.!?]+\s+", text.strip())

    # Take first N sentences
    summary_sentences = sentences[:max_sentences]
    summary = ". ".join(summary_sentences)

    # Add trailing period if missing
    if summary and not summary.endswith((".", "!", "?")):
        summary += "."

    return summary


def build_multifield_bm25_indexes(
    full_texts: list[str],
    headings: list[str],
    summaries: list[str],
    contextual_texts: list[str],
    node_ids: list[int],
) -> None:
    """
    Build and save four separate BM25 indexes for multi-field search.

    Args:
        full_texts: Complete chunk texts
        headings: Hierarchical heading contexts
        summaries: First 1-2 sentences of each chunk
        contextual_texts: LLM-generated contextual summaries + chunk texts
        node_ids: Corresponding node IDs for each document
    """
    settings = get_settings()
    index_path = Path(settings.bm25_index_path)
    index_path.mkdir(exist_ok=True)

    # Build four separate indexes
    _build_single_index(full_texts, node_ids, index_path / "full_text")
    _build_single_index(headings, node_ids, index_path / "headings")
    _build_single_index(summaries, node_ids, index_path / "summaries")
    _build_single_index(contextual_texts, node_ids, index_path / "contextual_text")

    # Save node_id mapping (shared across all indexes)
    import json

    mapping_file = index_path / "node_ids.json"
    mapping_file.write_text(json.dumps(node_ids))


def _build_single_index(
    corpus: list[str], node_ids: list[int], index_path: Path
) -> None:
    """
    Build and save a single BM25 index.

    Args:
        corpus: List of text documents
        node_ids: Corresponding node IDs
        index_path: Path to save the index
    """
    # 🤫 Tokenize corpus (silently - no tqdm progress bars)
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", show_progress=False)

    # Create and index BM25 retriever
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)

    # Save index
    index_path.mkdir(exist_ok=True, parents=True)
    retriever.save(str(index_path), corpus=corpus)
