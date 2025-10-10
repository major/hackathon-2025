"""Document processing using Docling with HybridChunker."""

import os
import tempfile
from pathlib import Path

import frontmatter
from sqlalchemy.orm import Session

from hackathon.database.node_ops import create_document_node
from hackathon.models.schemas import DocumentNodeCreate

# 🔇 Logging configured centrally in hackathon/__init__.py

# ⚡ LAZY IMPORTS: Docling is imported only when needed to avoid 2.3s startup delay
# These imports take ~2.3 seconds and are only needed during document processing,
# not during query operations. Moving them to function-level imports speeds up
# query startup from ~2.8s to ~0.5s!

# 🤗 Force HuggingFace offline mode (uses only cached models)
# Set this BEFORE any transformers imports to prevent network calls
# Requires bert-base-uncased to be cached in ~/.cache/huggingface/
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


def extract_yaml_frontmatter(
    file_path: Path,
) -> tuple[str, dict[str, str | int | float | bool]]:
    """
    Extract YAML frontmatter from markdown file.

    Args:
        file_path: Path to markdown file

    Returns:
        Tuple of (content without frontmatter, frontmatter dict)
    """
    post = frontmatter.load(str(file_path))
    # Keep original types (str, int, float, bool) rather than converting to str
    metadata: dict[str, str | int | float | bool] = {}
    for k, v in post.metadata.items():
        if isinstance(v, (str, int, float, bool)):
            metadata[str(k)] = v
        else:
            metadata[str(k)] = str(v)  # Fallback to string for other types
    return post.content, metadata


def process_document_with_docling(
    db: Session,
    document_id: int,
    file_path: Path,
    frontmatter: dict[str, str | int | float | bool],
) -> list[int]:
    """
    Process a document using Docling's HybridChunker.

    Args:
        db: Database session
        document_id: ID of the parent document
        file_path: Path to the document file
        frontmatter: YAML frontmatter metadata

    Returns:
        List of leaf node IDs
    """
    # Convert document with Docling
    doc = _convert_document_with_docling(file_path)

    # Get chunks using HybridChunker
    chunks = _chunk_document(doc)

    # Create database nodes from chunks (simple flat structure with positions)
    return _create_nodes_from_chunks(db, document_id, chunks, frontmatter)


def _convert_document_with_docling(file_path: Path):
    """Convert markdown document to Docling format, removing frontmatter first."""
    # ⚡ Lazy import: Only load Docling when actually converting documents
    from docling.document_converter import DocumentConverter

    clean_content, _ = extract_yaml_frontmatter(file_path)

    # Create temporary file with clean content
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as tmp_file:
        tmp_file.write(clean_content)
        tmp_path = Path(tmp_file.name)

    try:
        converter = DocumentConverter()
        result = converter.convert(str(tmp_path))
        return result.document
    finally:
        tmp_path.unlink(missing_ok=True)


def _chunk_document(doc):
    """Chunk a Docling document using HybridChunker in offline mode."""
    # ⚡ Lazy import: Only load HybridChunker when actually chunking documents
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

    # Offline mode is set at module level (see top of file)
    # This uses only cached models from ~/.cache/huggingface/
    chunker = HybridChunker(
        max_tokens=512,  # type: ignore
        tokenizer="bert-base-uncased",  # Must be pre-cached!
    )
    return list(chunker.chunk(doc))


def _fix_inline_code_blocks(text: str) -> str:
    """
    Fix Docling bug where inline code is formatted as fenced code blocks.

    Docling sometimes converts inline code like `term` to:
        ```
        term
        ```

    This function detects and fixes this pattern by converting back to inline code.

    Args:
        text: Chunk text possibly containing incorrectly formatted code blocks

    Returns:
        Text with inline code blocks fixed
    """
    import re

    # Pattern: ```\nsingle_line\n``` (with optional whitespace)
    # Matches code blocks that contain only a single line
    pattern = r"```\s*\n([^\n]+)\n```"

    def replacer(match):
        code_content = match.group(1).strip()
        # Only convert to inline code if it's short (likely an inline term)
        # and doesn't contain special characters that suggest it's real code
        if len(code_content) < 50 and "\n" not in code_content:
            return f"`{code_content}`"
        return match.group(0)  # Keep as-is if it looks like real code

    return re.sub(pattern, replacer, text)


def _extract_heading_context(chunk) -> str:
    """
    Extract heading context from a Docling chunk.

    Args:
        chunk: Docling chunk object

    Returns:
        Heading context as a string (e.g., "Section > Subsection")
    """
    if hasattr(chunk, "meta") and chunk.meta:
        if hasattr(chunk.meta, "headings") and chunk.meta.headings:
            return " > ".join(chunk.meta.headings)
    return ""


def _infer_node_type(chunk_text: str) -> str:
    """
    Infer node type from chunk content using heuristics.

    Args:
        chunk_text: Text content of the chunk

    Returns:
        Node type string ("code", "list", or "paragraph")
    """
    # Code heuristic: contains code blocks or starts with indentation
    if "```" in chunk_text or "    " in chunk_text[:20]:
        return "code"

    # List heuristic: starts with list markers
    if chunk_text.strip().startswith(("- ", "* ", "1. ", "2. ")):
        return "list"

    return "paragraph"


def _create_nodes_from_chunks(
    db: Session,
    document_id: int,
    chunks: list,
    frontmatter: dict[str, str | int | float | bool],
) -> list[int]:
    """
    Create database nodes from chunks with sequential position tracking.

    This creates a flat structure where each chunk is a node with a sequential position.
    The position field enables efficient neighbor retrieval (chunk N-1, N, N+1, etc.).

    Args:
        db: Database session
        document_id: ID of the parent document
        chunks: List of Docling chunk objects
        frontmatter: Document frontmatter metadata

    Returns:
        List of node IDs
    """
    node_ids = []
    position = 0

    for chunk in chunks:
        # Extract and clean chunk text
        chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
        chunk_text = _fix_inline_code_blocks(chunk_text)

        # Extract metadata from chunk
        headings = _extract_heading_context(chunk)
        node_type = _infer_node_type(chunk_text)

        # Build metadata combining frontmatter and chunk context
        metadata = {**frontmatter, "headings": headings}

        # Create node with flat structure (no parent relationships)
        chunk_node = DocumentNodeCreate(
            document_id=document_id,
            node_type=node_type,
            text_content=chunk_text,
            is_leaf=True,
            node_path=f"chunk_{position}",
            position=position,
            metadata=metadata,
        )

        db_node = create_document_node(db, chunk_node)
        _generate_tsvector(db, db_node)

        node_ids.append(db_node.id)
        position += 1

    return node_ids


def _generate_tsvector(db: Session, node) -> None:
    """
    Generate PostgreSQL tsvector for full-text search.

    Args:
        db: Database session
        node: DocumentNode instance
    """
    from sqlalchemy import text

    if node.text_content:
        # Use PostgreSQL's to_tsvector function to generate the search vector
        # Using 'english' dictionary for better stemming and stop word handling
        db.execute(
            text(
                "UPDATE document_nodes "
                "SET text_search = to_tsvector('english', :text_content) "
                "WHERE id = :node_id"
            ),
            {"text_content": node.text_content, "node_id": node.id},
        )
        db.commit()
