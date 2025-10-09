"""Query CLI command for testing the RAG system."""

import sys
import traceback

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sqlalchemy import select

from hackathon.database import get_db
from hackathon.models.database import DocumentNode
from hackathon.retrieval import ContextExpander, HybridSearcher

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
            semantic_block_text = expander.get_semantic_block_text(node)
            has_expanded_block = (
                semantic_block_text and semantic_block_text != node.text_content
            )

            context_text = expander.build_context_text(
                node,
                depth=2,
                include_siblings=False,
                exclude_current=has_expanded_block,
            )

            if has_expanded_block:
                full_context = f"[Semantic Block]\n{semantic_block_text}\n\n[Hierarchical Context]\n{context_text}"
                console.print(
                    Panel(
                        full_context,
                        title="Full Context (with Semantic Block)",
                        border_style="green",
                    )
                )
            else:
                console.print(
                    Panel(context_text, title="Full Context", border_style="green")
                )
    finally:
        db.close()


def main() -> None:
    """Main entry point for query command."""
    console.print("[bold blue]RAG System Query Interface[/bold blue]\n")

    # Get database session
    db_gen = get_db()
    db = next(db_gen)

    try:
        # Initialize searcher and expander
        searcher = HybridSearcher(db)
        expander = ContextExpander(db)

        # Get query from command line args or prompt
        if len(sys.argv) > 1:
            # Direct query mode (non-interactive)
            query = " ".join(sys.argv[1:])

            if not query.strip():
                console.print("[yellow]No query provided[/yellow]")
                return

            console.print(f"\n[green]Searching for:[/green] {query}\n")

            # Perform hybrid search
            results = searcher.hybrid_search(query, top_k=5)

            # Display results
            display_results(results, expander, expand_context=True)

            # Exit after showing results (non-interactive mode)
            return
        else:
            # Interactive mode
            query = console.input("[bold cyan]Enter your query:[/bold cyan] ")

            if not query.strip():
                console.print("[yellow]No query provided[/yellow]")
                return

            console.print(f"\n[green]Searching for:[/green] {query}\n")

            # Perform hybrid search
            results = searcher.hybrid_search(query, top_k=5)

            # Display results
            display_results(results, expander, expand_context=True)

            # Continue with interactive loop
            _interactive_loop(results, searcher, expander, db)

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        traceback.print_exc()
        sys.exit(1)
    finally:
        db.close()

    console.print("\n[bold blue]Goodbye![/bold blue]")


def _interactive_loop(results, searcher, expander, db):
    """Interactive loop for query interface."""
    while True:
        choice = _get_user_choice()

        if choice == "1":
            _handle_new_query(results, searcher, expander)
        elif choice == "2":
            _handle_expand_result(results, expander, db)
        elif choice == "3":
            break
        else:
            console.print("[yellow]Invalid choice[/yellow]")


def _get_user_choice() -> str:
    """Display menu and get user choice."""
    console.print("\n[bold]Options:[/bold]")
    console.print("1. New query")
    console.print("2. Show expanded context for a result")
    console.print("3. Exit")
    return console.input("[cyan]Choose an option (1-3):[/cyan] ")


def _handle_new_query(results, searcher, expander):
    """Handle new query from user."""
    query = console.input("[bold cyan]Enter your query:[/bold cyan] ")
    if query.strip():
        console.print(f"\n[green]Searching for:[/green] {query}\n")
        new_results = searcher.hybrid_search(query, top_k=5)
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


def _show_expanded_result(result, expander, db):
    """Show expanded context for a specific result."""
    stmt = select(DocumentNode).where(DocumentNode.id == result.node_id)
    node = db.execute(stmt).scalar_one_or_none()

    if node:
        semantic_block_text = expander.get_semantic_block_text(node)
        context_text = expander.build_context_text(node, depth=3, include_siblings=True)

        if semantic_block_text and semantic_block_text != node.text_content:
            full_context = f"[Semantic Block]\n{semantic_block_text}\n\n[Hierarchical Context]\n{context_text}"
            console.print(
                Panel(
                    full_context,
                    title="Expanded Context (with Semantic Block)",
                    border_style="green",
                )
            )
        else:
            console.print(
                Panel(context_text, title="Expanded Context", border_style="green")
            )


if __name__ == "__main__":
    main()
