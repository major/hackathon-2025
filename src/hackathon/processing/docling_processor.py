"""Document processing using Docling."""

import tempfile
from pathlib import Path

import frontmatter
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from hackathon.config import get_settings
from hackathon.database.operations import create_document_node
from hackathon.models.schemas import DocumentNodeCreate


def extract_yaml_frontmatter(file_path: Path) -> tuple[str, dict[str, str]]:
    """
    Extract YAML frontmatter from markdown file.

    Args:
        file_path: Path to markdown file

    Returns:
        Tuple of (content without frontmatter, frontmatter dict)
    """
    post = frontmatter.load(file_path)
    # Convert any non-string values to strings for consistency
    metadata = {str(k): str(v) for k, v in post.metadata.items()}
    return post.content, metadata


def process_document_with_docling(
    db: Session,
    document_id: int,
    file_path: Path,
    frontmatter: dict[str, str],
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

    # Create database nodes from chunks
    return _create_nodes_from_chunks(db, document_id, chunks, frontmatter)


def _convert_document_with_docling(file_path: Path):
    """Convert markdown document to Docling format, removing frontmatter first."""
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
    """Chunk a Docling document using HybridChunker with granite tokenizer."""
    settings = get_settings()
    model = SentenceTransformer(
        settings.embedding_model, device="cpu", trust_remote_code=True
    )
    # Configure chunker with max_tokens to prevent exceeding model limits
    # Use slightly less than 512 to account for special tokens
    chunker = HybridChunker(
        tokenizer=model.tokenizer,
        max_tokens=480,  # Leave headroom for [CLS] and [SEP] tokens
    )
    return list(chunker.chunk(doc))


def _create_nodes_from_chunks(
    db: Session, document_id: int, chunks: list, frontmatter: dict[str, str]
) -> list[int]:
    """Create database nodes from Docling chunks."""
    leaf_node_ids = []

    for chunk_counter, chunk in enumerate(chunks):
        node_type, chunk_metadata = _extract_chunk_metadata(chunk, frontmatter)

        chunk_node = DocumentNodeCreate(
            document_id=document_id,
            parent_id=None,
            node_type=node_type,
            text_content=chunk.text,
            is_leaf=True,
            node_path=f"chunk_{chunk_counter}",
            metadata=chunk_metadata,
        )

        db_node = create_document_node(db, chunk_node)
        leaf_node_ids.append(db_node.id)

    return leaf_node_ids


def _extract_chunk_metadata(chunk, frontmatter: dict[str, str]) -> tuple[str, dict]:
    """Extract node type and metadata from a Docling chunk."""
    meta = chunk.meta
    doc_items = getattr(meta, "doc_items", None) or []

    # Extract node type and item metadata
    node_type, docling_types, doc_item_refs, doc_item_parents = _process_doc_items(
        doc_items
    )

    # Build final metadata dictionary
    chunk_metadata = _build_metadata_dict(
        meta, frontmatter, docling_types, doc_item_refs, doc_item_parents
    )

    return node_type, chunk_metadata


def _process_doc_items(doc_items: list) -> tuple[str, list, list, list]:
    """Process Docling doc items to extract types and references."""
    node_type = "paragraph"
    docling_types = []
    doc_item_refs = []
    doc_item_parents = []

    for item in doc_items:
        if label := getattr(item, "label", None):
            label_str = str(label)
            docling_types.append(label_str)
            if node_type == "paragraph" and label_str != "text":
                node_type = label_str

        if self_ref := getattr(item, "self_ref", None):
            doc_item_refs.append(str(self_ref))

        if (parent := getattr(item, "parent", None)) and (
            parent_ref := getattr(parent, "cref", None)
        ):
            doc_item_parents.append(str(parent_ref))

    return node_type, docling_types, doc_item_refs, doc_item_parents


def _build_metadata_dict(
    meta,
    frontmatter: dict,
    docling_types: list,
    doc_item_refs: list,
    doc_item_parents: list,
) -> dict:
    """Build metadata dictionary from extracted information."""
    chunk_metadata = {**frontmatter}

    if headings := getattr(meta, "headings", None):
        chunk_metadata["headings"] = ", ".join(str(h) for h in headings)

    if docling_types:
        chunk_metadata["docling_types"] = ", ".join(docling_types)

    if doc_item_refs:
        chunk_metadata["doc_item_refs"] = ", ".join(doc_item_refs)

    if doc_item_parents:
        chunk_metadata["doc_item_parents"] = ", ".join(list(set(doc_item_parents)))

    if origin := getattr(meta, "origin", None):
        chunk_metadata["origin"] = str(origin)

    return chunk_metadata
