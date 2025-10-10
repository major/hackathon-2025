"""Query CLI command for testing the RAG system."""

import argparse
import sys
import traceback

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sqlalchemy import select

from hackathon.database import get_db
from hackathon.models.database import DocumentNode
from hackathon.retrieval import ContextExpander
from hackathon.retrieval.multifield_searcher import MultiFieldBM25Searcher

console = Console()

PREVIEW_LENGTH = 200


def display_results(results, expander=None, expand_context: bool = False):
    """
    Display search results in a formatted table.

    Args:
        results: List of SearchResult objects
        expander: Optional ContextExpander for context expansion
        expand_context: Whether to show expanded context
    """
    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    _display_results_table(results, expander)

    if expand_context and results and expander:
        _display_expanded_context(results[0], expander)


def _display_results_table(results, expander):
    """Display search results in a formatted table."""
    table = Table(title="Search Results", show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Score", style="green", width=8)
    table.add_column("Type", style="blue", width=12)
    table.add_column("Path", style="yellow", width=15)
    table.add_column("Content Preview", style="white", width=60)

    db_gen = get_db()
    db = next(db_gen)

    try:
        for idx, result in enumerate(results, 1):
            display_text = _get_display_text(db, result, expander)
            preview = _create_preview(display_text)
            table.add_row(
                str(idx),
                f"{result.score:.4f}",
                result.node_type,
                result.node_path,
                preview,
            )
    finally:
        db.close()

    console.print(table)


def _get_display_text(db, result, expander):
    """Get display text for a result, expanding semantic blocks if available."""
    if not expander:
        return result.text_content

    stmt = select(DocumentNode).where(DocumentNode.id == result.node_id)
    node = db.execute(stmt).scalar_one_or_none()

    if node:
        semantic_block_text = expander.get_semantic_block_text(node)
        if semantic_block_text and semantic_block_text != node.text_content:
            return semantic_block_text

    return result.text_content


def _create_preview(text: str) -> str:
    """Create a preview of text with ellipsis if needed."""
    return text[:PREVIEW_LENGTH] + "..." if len(text) > PREVIEW_LENGTH else text


def _display_expanded_context(top_result, expander):
    """Display expanded context for the top search result."""
    console.print("\n[bold]Expanded Context for Top Result:[/bold]")

    db_gen = get_db()
    db = next(db_gen)

    try:
        stmt = select(DocumentNode).where(DocumentNode.id == top_result.node_id)
        node = db.execute(stmt).scalar_one_or_none()

        if node:
            # Get semantic block if available
            semantic_block_text = expander.get_semantic_block_text(node)
            has_expanded_block = (
                semantic_block_text and semantic_block_text != node.text_content
            )

            # Build context parts
            context_parts = []

            # Add document context (metadata)
            metadata = node.meta or {}
            if headings := metadata.get("headings"):
                context_parts.append(
                    f"[bold cyan]Document Section:[/bold cyan] {headings}"
                )
            if title := metadata.get("title"):
                context_parts.append(f"[bold cyan]Document Title:[/bold cyan] {title}")
            if date := metadata.get("date"):
                context_parts.append(f"[bold cyan]Date:[/bold cyan] {date}")

            # Add node type
            context_parts.append(
                f"\n[bold cyan]Content Type:[/bold cyan] {node.node_type}"
            )

            # Add the actual content
            if has_expanded_block:
                context_parts.append(
                    f"\n[bold cyan]Full Content:[/bold cyan]\n{semantic_block_text}"
                )
            else:
                context_parts.append(
                    f"\n[bold cyan]Content:[/bold cyan]\n{node.text_content}"
                )

            full_context = "\n".join(context_parts)
            console.print(
                Panel(
                    full_context,
                    title="Full Context",
                    border_style="green",
                )
            )
    finally:
        db.close()


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Query the RAG system with configurable context expansion"
    )
    parser.add_argument(
        "query", nargs="*", help="Search query (interactive if omitted)"
    )
    parser.add_argument(
        "--expand",
        "-e",
        type=int,
        default=None,
        metavar="LEVELS",
        help="Context expansion: 0=node only, N=N chunks before+after (e.g., 2=2 before + 2 after), -1=full document",
    )
    parser.add_argument(
        "--neighbors",
        "-n",
        type=str,
        default=None,
        metavar="BEFORE,AFTER",
        help="Neighbor expansion: get N neighbors before and after (e.g., '2,2' or '1,3'). Use single number for symmetric (e.g., '2' = 2 before + 2 after)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        metavar="K",
        help="Number of top results to return (default: 5)",
    )
    parser.add_argument(
        "--rerank",
        "-r",
        action="store_true",
        help="Use IBM Watsonx semantic reranking for improved relevance (requires WATSONX_API_KEY and WATSONX_PROJECT_ID)",
    )
    parser.add_argument(
        "--candidates",
        "-c",
        type=int,
        default=50,
        metavar="N",
        help="Number of candidates to retrieve before reranking (default: 50, only used with --rerank)",
    )
    return parser


def _execute_query_and_display(query: str, searcher, expander, db, args) -> list:
    """Execute a query and display results with optional expansions."""
    console.print(f"\n[green]Searching for:[/green] {query}")
    if args.rerank:
        console.print(
            f"[cyan]🎯 Reranking enabled (candidates={args.candidates})[/cyan]\n"
        )
    else:
        console.print()

    # Perform multi-field BM25 search (with optional reranking)
    results = searcher.search(
        query,
        top_k=args.top_k,
        use_reranker=args.rerank,
        rerank_candidates=args.candidates,
    )

    # Display results
    display_results(results, expander, expand_context=True)

    # If --expand specified, show expanded context for top result
    if args.expand is not None and results:
        _show_expanded_result_with_level(results[0], expander, db, args.expand)

    # If --neighbors specified, show neighbor context for top result
    if args.neighbors and results:
        _show_neighbor_expansion(results[0], expander, db, args.neighbors)

    return results


def main() -> None:
    """Main entry point for query command."""
    args = _create_argument_parser().parse_args()

    console.print("[bold blue]RAG System Query Interface[/bold blue]\n")

    # Get database session
    db_gen = get_db()
    db = next(db_gen)

    try:
        # ⚡ Initialize searcher and expander ONCE (indexes loaded into memory here ~1-2 sec)
        console.print("[yellow]Loading BM25 indexes...[/yellow]")
        searcher = MultiFieldBM25Searcher(db)
        expander = ContextExpander(db)
        console.print("[green]✓ Indexes loaded![/green]")

        # Get query from command line args or interactive prompt
        query = (
            " ".join(args.query)
            if args.query
            else console.input("[bold cyan]Enter your query:[/bold cyan] ")
        )

        if not query.strip():
            console.print("[yellow]No query provided[/yellow]")
            return

        # Execute query and display results
        results = _execute_query_and_display(query, searcher, expander, db, args)

        # If direct query mode (non-interactive), exit after showing results
        if args.query:
            return

        # Otherwise, continue with interactive loop
        _interactive_loop(results, searcher, expander, db, args)

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        traceback.print_exc()
        sys.exit(1)
    finally:
        db.close()

    console.print("\n[bold blue]Goodbye![/bold blue]")


def _interactive_loop(results, searcher, expander, db, args):
    """Interactive loop for query interface."""
    while True:
        choice = _get_user_choice()

        if choice == "1":
            _handle_new_query(results, searcher, expander, args)
        elif choice == "2":
            _handle_expand_result(results, expander, db)
        elif choice == "3":
            _handle_neighbor_expansion(results, expander, db)
        elif choice == "4":
            break
        else:
            console.print("[yellow]Invalid choice[/yellow]")


def _get_user_choice() -> str:
    """Display menu and get user choice."""
    console.print("\n[bold]Options:[/bold]")
    console.print("1. New query")
    console.print("2. Expand context for a result (hierarchical/parent levels)")
    console.print("3. Expand with neighbors (document order)")
    console.print("4. Exit")
    return console.input("[cyan]Choose an option (1-4):[/cyan] ")


def _handle_new_query(results, searcher, expander, args):
    """Handle new query from user."""
    query = console.input("[bold cyan]Enter your query:[/bold cyan] ")
    if query.strip():
        console.print(f"\n[green]Searching for:[/green] {query}")
        if args.rerank:
            console.print("[cyan]🎯 Reranking enabled[/cyan]\n")
        else:
            console.print()
        new_results = searcher.search(
            query,
            top_k=args.top_k,
            use_reranker=args.rerank,
            rerank_candidates=args.candidates,
        )
        display_results(new_results, expander, expand_context=False)
        results[:] = new_results


def _handle_expand_result(results, expander, db):
    """Handle expanding a specific result."""
    if not results:
        console.print("[yellow]No results to expand[/yellow]")
        return

    result_num = console.input(f"[cyan]Enter result number (1-{len(results)}):[/cyan] ")

    try:
        idx = int(result_num) - 1
        if 0 <= idx < len(results):
            _show_expanded_result(results[idx], expander, db)
        else:
            console.print("[red]Invalid result number[/red]")
    except ValueError:
        console.print("[red]Invalid input[/red]")


def _handle_neighbor_expansion(results, expander, db):
    """Handle expanding a result with neighbors."""
    if not results:
        console.print("[yellow]No results to expand[/yellow]")
        return

    result_num = console.input(f"[cyan]Enter result number (1-{len(results)}):[/cyan] ")

    try:
        idx = int(result_num) - 1
        if 0 <= idx < len(results):
            # Get neighbor counts from user
            neighbor_input = console.input(
                "[cyan]Enter neighbor count (e.g., '2' for 2 before+after, or '2,3' for 2 before + 3 after):[/cyan] "
            )
            _show_neighbor_expansion(results[idx], expander, db, neighbor_input)
        else:
            console.print("[red]Invalid result number[/red]")
    except ValueError:
        console.print("[red]Invalid input[/red]")


def _show_expanded_result(result, expander, db):
    """Show expanded context for a specific result with configurable levels."""
    stmt = select(DocumentNode).where(DocumentNode.id == result.node_id)
    node = db.execute(stmt).scalar_one_or_none()

    if not node:
        console.print("[red]Node not found[/red]")
        return

    # Get size estimates for different expansion levels
    sizes = expander.estimate_context_sizes(node, max_levels=5)

    # Show expansion options
    console.print("\n[bold cyan]Context Expansion Options:[/bold cyan]")
    console.print(f"0. Node only ({_format_size(sizes.get('node', 0))})")

    # Show parent levels
    level_count = 0
    for key in sorted(sizes.keys()):
        if key.startswith("level_"):
            level_num = key.split("_")[1]
            level_count += 1
            size = sizes[key]
            console.print(
                f"{level_count}. Parent level {level_num} ({_format_size(size)})"
            )

    # Show full document option
    full_doc_option = level_count + 1
    console.print(
        f"{full_doc_option}. Full document (MAX) ({_format_size(sizes.get('full_document', 0))})"
    )

    # Get user's choice
    choice = console.input(
        f"[cyan]Choose expansion level (0-{full_doc_option}):[/cyan] "
    )

    try:
        level = int(choice)
        if level < 0 or level > full_doc_option:
            console.print("[red]Invalid choice[/red]")
            return

        # Generate context based on choice
        if level == 0:
            # Node only
            context_text = _format_node_only_context(node, expander)
        elif level == full_doc_option:
            # Full document
            context_text = expander.get_full_document_context(node)
        else:
            # Parent level expansion
            context_text = expander.get_parent_context(node, levels=level)

        # Display the expanded context
        _display_context_with_metadata(node, context_text, level, full_doc_option)

    except ValueError:
        console.print("[red]Invalid input[/red]")


def _format_size(size: int) -> str:
    """Format size with character count and approximate token estimate."""
    # Rough estimate: ~4 characters per token for English text
    tokens = size // 4
    if size < 1000:
        return f"{size} chars, ~{tokens} tokens"
    elif size < 1_000_000:
        return f"{size / 1000:.1f}K chars, ~{tokens:,} tokens"
    else:
        return f"{size / 1_000_000:.1f}M chars, ~{tokens:,} tokens"


def _format_node_only_context(node: DocumentNode, expander: ContextExpander) -> str:
    """Format just the node content (with semantic block expansion if applicable)."""
    semantic_block_text = expander.get_semantic_block_text(node)
    if semantic_block_text and semantic_block_text != node.text_content:
        return expander._format_semantic_block(node, semantic_block_text)
    return expander._format_node_content(node, "current")


def _display_context_with_metadata(
    node: DocumentNode, context_text: str, level: int, full_doc_option: int
) -> None:
    """Display expanded context with metadata header."""
    metadata = node.meta or {}
    header_parts = []

    # Add metadata
    if title := metadata.get("title"):
        header_parts.append(f"[bold cyan]Title:[/bold cyan] {title}")
    if headings := metadata.get("headings"):
        header_parts.append(f"[bold cyan]Section:[/bold cyan] {headings}")
    if date := metadata.get("date"):
        header_parts.append(f"[bold cyan]Date:[/bold cyan] {date}")

    # Add expansion level indicator
    if level == 0:
        expansion_desc = "Node Only"
    elif level == full_doc_option:
        expansion_desc = "Full Document"
    else:
        expansion_desc = f"Parent Level {level}"

    header_parts.append(f"[bold cyan]Expansion:[/bold cyan] {expansion_desc}")
    header_parts.append(
        f"[bold cyan]Size:[/bold cyan] {_format_size(len(context_text))}"
    )

    # Combine header and content
    full_output = "\n".join(header_parts) + "\n\n" + context_text

    console.print(
        Panel(
            full_output,
            title="Expanded Context",
            border_style="green",
        )
    )


def _parse_neighbor_args(neighbor_str: str) -> tuple[int, int]:
    """
    Parse the --neighbors argument.

    Args:
        neighbor_str: String like "2", "2,2", or "1,3"

    Returns:
        Tuple of (before, after) counts

    Raises:
        ValueError: If the format is invalid
    """
    if "," in neighbor_str:
        parts = neighbor_str.split(",")
        if len(parts) != 2:
            raise ValueError("Neighbor format must be 'N' or 'BEFORE,AFTER'")
        before = int(parts[0].strip())
        after = int(parts[1].strip())
    else:
        # Symmetric: same number before and after
        count = int(neighbor_str.strip())
        before = after = count

    if before < 0 or after < 0:
        raise ValueError("Neighbor counts must be non-negative")

    return before, after


def _show_neighbor_expansion(
    result, expander: ContextExpander, db, neighbor_str: str
) -> None:
    """
    Show neighbor context for a result.

    Args:
        result: SearchResult object
        expander: ContextExpander instance
        db: Database session
        neighbor_str: String specifying neighbor counts (e.g., "2" or "2,3")
    """
    stmt = select(DocumentNode).where(DocumentNode.id == result.node_id)
    node = db.execute(stmt).scalar_one_or_none()

    if not node:
        console.print("[red]Node not found[/red]")
        return

    try:
        before, after = _parse_neighbor_args(neighbor_str)

        # Generate neighbor context
        context_text = expander.get_neighbor_context(node, before=before, after=after)

        # Display the context
        metadata = node.meta or {}
        header_parts = []

        # Add metadata
        if title := metadata.get("title"):
            header_parts.append(f"[bold cyan]Title:[/bold cyan] {title}")
        if headings := metadata.get("headings"):
            header_parts.append(f"[bold cyan]Section:[/bold cyan] {headings}")
        if date := metadata.get("date"):
            header_parts.append(f"[bold cyan]Date:[/bold cyan] {date}")

        header_parts.append(
            f"[bold cyan]Expansion:[/bold cyan] Neighbors ({before} before, {after} after)"
        )
        header_parts.append(
            f"[bold cyan]Size:[/bold cyan] {_format_size(len(context_text))}"
        )

        # Combine header and content
        full_output = "\n".join(header_parts) + "\n\n" + context_text

        console.print(
            Panel(
                full_output,
                title="Neighbor Context",
                border_style="cyan",
            )
        )

    except ValueError as e:
        console.print(f"[red]Invalid neighbor format: {e}[/red]")


def _show_expanded_result_with_level(
    result, expander: ContextExpander, db, level: int
) -> None:
    """
    Show expanded context for a result with a specific level.

    Args:
        result: SearchResult object
        expander: ContextExpander instance
        db: Database session
        level: Expansion level (-1 for max, 0 for node only, N for N parent levels)
    """
    stmt = select(DocumentNode).where(DocumentNode.id == result.node_id)
    node = db.execute(stmt).scalar_one_or_none()

    if not node:
        console.print("[red]Node not found[/red]")
        return

    # Generate context based on level
    if level == -1:
        # Max expansion = full document
        context_text = expander.get_full_document_context(node)
        expansion_desc = "Full Document (MAX)"
    elif level == 0:
        # Node only
        context_text = _format_node_only_context(node, expander)
        expansion_desc = "Node Only"
    else:
        # Parent level expansion
        context_text = expander.get_parent_context(node, levels=level)
        expansion_desc = f"Parent Level {level}"

    # Display the expanded context
    metadata = node.meta or {}
    header_parts = []

    # Add metadata
    if title := metadata.get("title"):
        header_parts.append(f"[bold cyan]Title:[/bold cyan] {title}")
    if headings := metadata.get("headings"):
        header_parts.append(f"[bold cyan]Section:[/bold cyan] {headings}")
    if date := metadata.get("date"):
        header_parts.append(f"[bold cyan]Date:[/bold cyan] {date}")

    header_parts.append(f"[bold cyan]Expansion:[/bold cyan] {expansion_desc}")
    header_parts.append(
        f"[bold cyan]Size:[/bold cyan] {_format_size(len(context_text))}"
    )

    # Combine header and content
    full_output = "\n".join(header_parts) + "\n\n" + context_text

    console.print(
        Panel(
            full_output,
            title="Expanded Context",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
