"""Document processing CLI command."""

import sys
import traceback
from pathlib import Path

from rich.console import Console
from rich.progress import track

from hackathon.config import get_settings
from hackathon.database import create_document, get_all_leaf_nodes, get_db, init_db
from hackathon.models.schemas import DocumentCreate
from hackathon.processing import (
    ContextualRetriever,
    EmbeddingGenerator,
)
from hackathon.processing.bm25 import create_bm25_index_for_node
from hackathon.processing.docling_processor import (
    extract_yaml_frontmatter,
    process_document_with_docling,
)

console = Console()


def find_markdown_files(directory: Path, pattern: str) -> list[Path]:
    """
    Find all markdown files matching the pattern.

    Args:
        directory: Base directory to search
        pattern: Glob pattern for matching files

    Returns:
        List of markdown file paths
    """
    return sorted(directory.glob(pattern))


def process_single_document(db_session, md_path: Path) -> int:
    """
    Process a single document using Docling.

    Args:
        db_session: Database session
        md_path: Path to document file

    Returns:
        Document ID
    """
    # Extract YAML frontmatter if present
    _, frontmatter = extract_yaml_frontmatter(md_path)

    # Create document entry - use full path as filename for uniqueness
    doc_data = DocumentCreate(
        filename=str(md_path),
        filepath=str(md_path),
        metadata=frontmatter,
    )
    document = create_document(db_session, doc_data)

    # Process document with Docling
    process_document_with_docling(
        db=db_session,
        document_id=document.id,
        file_path=md_path,
        frontmatter=frontmatter,
    )

    return document.id


def main() -> None:
    """Main entry point for document processing."""
    console.print("[bold blue]Starting document processing...[/bold blue]")

    # Initialize and validate
    settings = get_settings()
    init_db()
    md_files = _get_markdown_files(settings)

    # Process all documents
    db_gen = get_db()
    db = next(db_gen)

    try:
        document_ids = _process_documents(db, md_files)
        all_leaf_nodes = _collect_leaf_nodes(db, document_ids)

        _build_bm25_index(db, all_leaf_nodes)
        _generate_embeddings(db, all_leaf_nodes)
        _generate_contextual_summaries(db, document_ids, all_leaf_nodes)

        console.print("\n[bold green]Processing complete![/bold green]")
        console.print(
            f"Processed {len(md_files)} documents with {len(all_leaf_nodes)} chunks"
        )

    except Exception as e:
        console.print(f"[bold red]Error during processing: {e}[/bold red]")
        traceback.print_exc()
        sys.exit(1)
    finally:
        db.close()


def _get_markdown_files(settings) -> list[Path]:
    """Find and validate markdown files."""
    console.print("[yellow]Initializing database...[/yellow]")

    md_directory = Path(settings.markdown_directory)
    if not md_directory.exists():
        console.print(f"[red]Error: Directory not found: {md_directory}[/red]")
        sys.exit(1)

    md_files = find_markdown_files(md_directory, settings.markdown_pattern)
    if not md_files:
        console.print(f"[yellow]No markdown files found in {md_directory}[/yellow]")
        sys.exit(0)

    console.print(f"[green]Found {len(md_files)} markdown files[/green]")
    return md_files


def _process_documents(db, md_files: list[Path]) -> list[int]:
    """Process all markdown files and return document IDs."""
    console.print("\n[bold]Step 1: Parsing and chunking documents[/bold]")
    document_ids = []

    for md_file in track(md_files, description="Processing files"):
        doc_id = process_single_document(db, md_file)
        document_ids.append(doc_id)

    return document_ids


def _collect_leaf_nodes(db, document_ids: list[int]) -> list:
    """Collect all leaf nodes from documents."""
    all_leaf_nodes = []
    for doc_id in document_ids:
        leaf_nodes = get_all_leaf_nodes(db, doc_id)
        all_leaf_nodes.extend(leaf_nodes)

    console.print(f"[green]Created {len(all_leaf_nodes)} chunks[/green]")
    return all_leaf_nodes


def _build_bm25_index(db, all_leaf_nodes: list) -> None:
    """Build BM25 index for all nodes."""
    console.print("\n[bold]Step 2: Building BM25 index[/bold]")
    for node in track(all_leaf_nodes, description="BM25 indexing"):
        if node.text_content:
            create_bm25_index_for_node(db, node.id, node.text_content)
    db.commit()


def _generate_embeddings(db, all_leaf_nodes: list) -> None:
    """Generate embeddings for all nodes."""
    console.print("\n[bold]Step 3: Generating embeddings[/bold]")
    embedder = EmbeddingGenerator()

    node_texts = [
        (node.id, node.text_content or "")
        for node in all_leaf_nodes
        if node.text_content
    ]

    batches = [
        node_texts[i : i + embedder.batch_size]
        for i in range(0, len(node_texts), embedder.batch_size)
    ]

    for batch in track(batches, description="Embedding batches"):
        embedder.batch_create_embeddings(db, batch)


def _generate_contextual_summaries(
    db, document_ids: list[int], all_leaf_nodes: list
) -> None:
    """Generate contextual summaries for all nodes."""
    console.print("\n[bold]Step 4: Generating contextual summaries[/bold]")
    contextual = ContextualRetriever()

    for doc_id in document_ids:
        node_objects = [node for node in all_leaf_nodes if node.document_id == doc_id]

        for node in track(
            node_objects, description=f"Contextual processing (doc {doc_id})"
        ):
            contextual.create_contextual_chunk_for_node(db, node, document_context="")


if __name__ == "__main__":
    main()
