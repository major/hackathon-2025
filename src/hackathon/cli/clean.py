"""Database cleanup CLI command."""

from rich.console import Console

from hackathon.database import reset_db

console = Console()


def main() -> None:
    """Drop all database tables and recreate them."""
    console.print("[yellow]Dropping all tables...[/yellow]")
    reset_db()
    console.print("[bold green]Database reset complete![/bold green]")


if __name__ == "__main__":
    main()
