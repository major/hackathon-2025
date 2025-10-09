"""Centralized logging configuration using rich."""

import logging

from rich.console import Console
from rich.logging import RichHandler

# Global console instance
console = Console()

# Log level configuration
LOG_LEVEL = logging.WARNING


def setup_logging(level: int = LOG_LEVEL) -> logging.Logger:
    """
    Configure centralized logging with rich formatting.

    Args:
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance

    Example:
        >>> from hackathon.utils.logging import setup_logging
        >>> logger = setup_logging()
        >>> logger.info("Processing document")
    """
    # Create rich handler
    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        show_time=True,
        show_path=True,
    )

    # Configure format
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=[handler],
        force=True,  # Override any existing configuration
    )

    # Return logger
    return logging.getLogger("hackathon")


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (default: "hackathon")

    Returns:
        Logger instance

    Example:
        >>> from hackathon.utils.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting process")
    """
    if name is None:
        name = "hackathon"
    return logging.getLogger(name)


# Initialize logging on module import
setup_logging()
