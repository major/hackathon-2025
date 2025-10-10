#!/usr/bin/env python3
"""Download and cache the BERT tokenizer for offline use.

This script downloads bert-base-uncased tokenizer to ~/.cache/huggingface/
so that document processing can run in offline mode.

Run this once before using the processing pipeline:
    uv run download-tokenizer
"""

from rich.console import Console

console = Console()


def main():
    """Download BERT tokenizer for offline caching."""
    console.print(
        "[bold blue]Downloading BERT tokenizer for offline use...[/bold blue]"
    )

    try:
        from transformers import AutoTokenizer

        console.print("[yellow]Downloading bert-base-uncased (~500MB)...[/yellow]")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Verify it works
        test_tokens = tokenizer.tokenize("test")
        console.print("[green]✓ Tokenizer downloaded and cached![/green]")
        console.print("[green]  Location: ~/.cache/huggingface/hub/[/green]")
        console.print(
            f"[green]  Test tokenization: 'test' → {len(test_tokens)} tokens[/green]"
        )

        console.print(
            "\n[bold green]Success![/bold green] You can now run processing in offline mode:"
        )
        console.print("  [cyan]uv run process --skip-contextual[/cyan]")

    except Exception as e:
        console.print(f"[bold red]Error downloading tokenizer: {e}[/bold red]")
        console.print(
            "[yellow]Make sure you have internet connection for the first download.[/yellow]"
        )
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
