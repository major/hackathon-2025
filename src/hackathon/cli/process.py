"""Document processing CLI command."""

import argparse
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, track

from hackathon.config import get_settings
from hackathon.database import (
    create_document,
    create_multifield_bm25_index,
    get_all_leaf_nodes,
    get_db,
    get_document_by_id,
    init_db,
)
from hackathon.models.schemas import DocumentCreate, MultiFieldBM25IndexCreate
from hackathon.processing.bm25_indexing import (
    build_multifield_bm25_indexes,
    extract_summary,
)
from hackathon.processing.contextual import (
    generate_contextual_summary,
    generate_contextual_text,
)
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process markdown documents for RAG system (multi-field BM25 with optional contextual retrieval)"
    )
    parser.add_argument(
        "--skip-contextual",
        action="store_true",
        help="Skip contextual summary generation (faster, no LLM calls)",
    )
    args = parser.parse_args()

    console.print("[bold blue]Starting document processing...[/bold blue]")
    if args.skip_contextual:
        console.print(
            "[yellow]Skipping contextual summaries (--skip-contextual flag)[/yellow]"
        )
    else:
        console.print(
            "[cyan]Generating contextual summaries using IBM Watsonx Granite[/cyan]"
        )

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

        _build_multifield_bm25_indexes(
            db, all_leaf_nodes, skip_contextual=args.skip_contextual
        )

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
    """Process all markdown files in parallel and return document IDs."""
    console.print("\n[bold]Step 1: Parsing and chunking documents 🦆[/bold]")
    document_ids = []

    # Use ThreadPoolExecutor for I/O-bound Docling operations
    # Each thread gets its own database session (SQLAlchemy sessions are not thread-safe)
    max_workers = min(4, len(md_files))  # Cap at 4 workers to avoid overwhelming system

    with Progress() as progress:
        task = progress.add_task("🦆 Processing files", total=len(md_files))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files for processing - each worker creates its own db session
            future_to_file = {
                executor.submit(_process_document_worker, md_file): md_file
                for md_file in md_files
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                md_file = future_to_file[future]
                try:
                    doc_id = future.result()
                    document_ids.append(doc_id)
                    progress.advance(task)
                except Exception as e:
                    console.print(f"[red]Error processing {md_file}: {e}[/red]")
                    raise

    return document_ids


def _process_document_worker(md_file: Path) -> int:
    """
    Worker function for parallel document processing.

    Each worker creates its own database session (SQLAlchemy sessions are not thread-safe).
    """
    from hackathon.database import get_db

    db_gen = get_db()
    db = next(db_gen)
    try:
        return process_single_document(db, md_file)
    finally:
        db.close()


def _collect_leaf_nodes(db, document_ids: list[int]) -> list:
    """Collect all leaf nodes from documents."""
    all_leaf_nodes = []
    for doc_id in document_ids:
        leaf_nodes = get_all_leaf_nodes(db, doc_id)
        all_leaf_nodes.extend(leaf_nodes)

    console.print(f"[green]🦆 Created {len(all_leaf_nodes)} chunks[/green]")
    return all_leaf_nodes


def _build_multifield_bm25_indexes(
    db, all_leaf_nodes: list, skip_contextual: bool = False
) -> None:
    """Build multi-field BM25 indexes (full_text, headings, summary, contextual_text)."""
    console.print("\n[bold]Step 2: Building multi-field BM25 indexes[/bold]")

    # Prepare data for all four fields
    full_texts = []
    headings_list = []
    summaries = []
    contextual_texts = []
    node_ids = []

    console.print("[yellow]Preparing multi-field data...[/yellow]")

    if skip_contextual:
        # Sequential processing when no LLM calls needed
        for node in track(all_leaf_nodes, description="Processing nodes"):
            if not node.text_content:
                continue

            full_text = node.text_content
            metadata = node.meta or {}
            headings = metadata.get("headings", "") or f"{node.node_type}"
            summary = extract_summary(full_text, max_sentences=2)
            contextual_text = full_text  # No LLM call

            # Store in database
            index_data = MultiFieldBM25IndexCreate(
                node_id=node.id,
                full_text=full_text,
                headings=headings,
                summary=summary,
                contextual_text=contextual_text,
            )
            create_multifield_bm25_index(db, index_data)

            # Add to lists
            full_texts.append(full_text)
            headings_list.append(headings)
            summaries.append(summary)
            contextual_texts.append(contextual_text)
            node_ids.append(node.id)
    else:
        # Parallel processing for LLM calls (I/O-bound)
        _build_indexes_with_parallel_llm(
            db,
            all_leaf_nodes,
            full_texts,
            headings_list,
            summaries,
            contextual_texts,
            node_ids,
        )

    # Build and save persistent bm25s indexes
    num_fields = 3 if skip_contextual else 4
    console.print(
        f"[yellow]Building persistent BM25 indexes ({num_fields} fields)...[/yellow]"
    )
    build_multifield_bm25_indexes(
        full_texts, headings_list, summaries, contextual_texts, node_ids
    )
    console.print("[green]Multi-field BM25 indexes saved[/green]")


def _build_indexes_with_parallel_llm(
    db,
    all_leaf_nodes: list,
    full_texts: list,
    headings_list: list,
    summaries: list,
    contextual_texts: list,
    node_ids: list,
) -> None:
    """Build indexes with parallel LLM calls for contextual summaries."""
    # Prepare node data for parallel processing
    nodes_data = []
    for node in all_leaf_nodes:
        if not node.text_content:
            continue

        metadata = node.meta or {}
        headings = metadata.get("headings", "") or f"{node.node_type}"

        # Get document filename
        document = get_document_by_id(db, node.document_id)
        filename = Path(document.filename).name if document else "unknown"

        nodes_data.append({
            "node": node,
            "full_text": node.text_content,
            "headings": headings,
            "filename": filename,
        })

    # Use ThreadPoolExecutor for I/O-bound LLM API calls
    # Higher worker count is OK for I/O-bound tasks
    max_workers = min(10, len(nodes_data))

    with Progress() as progress:
        task = progress.add_task(
            "🤖 Generating contextual summaries", total=len(nodes_data)
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all LLM calls in parallel
            future_to_node = {
                executor.submit(_generate_contextual_for_node, node_data): node_data
                for node_data in nodes_data
            }

            # Collect results as they complete
            for future in as_completed(future_to_node):
                try:
                    result = future.result()

                    # Store in database
                    index_data = MultiFieldBM25IndexCreate(
                        node_id=result["node_id"],
                        full_text=result["full_text"],
                        headings=result["headings"],
                        summary=result["summary"],
                        contextual_text=result["contextual_text"],
                    )
                    create_multifield_bm25_index(db, index_data)

                    # Add to lists
                    full_texts.append(result["full_text"])
                    headings_list.append(result["headings"])
                    summaries.append(result["summary"])
                    contextual_texts.append(result["contextual_text"])
                    node_ids.append(result["node_id"])

                    progress.advance(task)
                except Exception as e:
                    console.print(
                        f"[red]Error generating contextual summary: {e}[/red]"
                    )
                    raise


def _generate_contextual_for_node(node_data: dict) -> dict:
    """
    Worker function for parallel contextual summary generation.

    This runs in a thread and makes LLM API calls.
    """
    node = node_data["node"]
    full_text = node_data["full_text"]
    headings = node_data["headings"]
    filename = node_data["filename"]

    # Generate summary (first 2 sentences)
    summary = extract_summary(full_text, max_sentences=2)

    # Build document context for LLM
    doc_context = {
        "filename": filename,
        "headings": headings,
    }

    # Generate contextual summary using IBM Watsonx (API call)
    contextual_summary = generate_contextual_summary(full_text, doc_context)
    contextual_text = generate_contextual_text(full_text, contextual_summary)

    return {
        "node_id": node.id,
        "full_text": full_text,
        "headings": headings,
        "summary": summary,
        "contextual_text": contextual_text,
    }


if __name__ == "__main__":
    main()
