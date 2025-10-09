"""Markdown parsing with YAML frontmatter removal."""

import re
from pathlib import Path


class MarkdownParser:
    """Markdown parser that removes YAML frontmatter."""

    def __init__(self) -> None:
        """Initialize the markdown parser."""

    @staticmethod
    def remove_frontmatter(content: str) -> tuple[str, dict[str, str]]:
        """
        Remove YAML frontmatter from markdown content.

        Args:
            content: Raw markdown content with potential frontmatter

        Returns:
            Tuple of (content without frontmatter, frontmatter dict)
        """
        frontmatter = {}

        # Pattern to match YAML frontmatter (between --- delimiters)
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if match:
            # Extract frontmatter
            fm_content = match.group(1)

            # Simple YAML parsing (key: value pairs)
            for line in fm_content.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    frontmatter[key.strip()] = value.strip()

            # Remove frontmatter from content
            content = content[match.end() :]

        return content.strip(), frontmatter

    def parse_markdown(self, md_path: str | Path) -> dict:
        """
        Parse a markdown file and remove YAML frontmatter.

        Args:
            md_path: Path to the markdown file

        Returns:
            Dictionary containing content and metadata

        Raises:
            FileNotFoundError: If the markdown file does not exist
        """
        md_path = Path(md_path)
        if not md_path.exists():
            msg = f"Markdown file not found: {md_path}"
            raise FileNotFoundError(msg)

        # Read the file
        with md_path.open(encoding="utf-8") as f:
            raw_content = f.read()

        # Remove frontmatter
        content, frontmatter = self.remove_frontmatter(raw_content)

        return {
            "content": content,
            "frontmatter": frontmatter,
            "filepath": str(md_path),
        }

    def extract_text(self, md_path: str | Path) -> str:
        """
        Extract text content from a markdown file (without frontmatter).

        Args:
            md_path: Path to the markdown file

        Returns:
            Extracted text content

        Raises:
            FileNotFoundError: If the markdown file does not exist
        """
        parsed = self.parse_markdown(md_path)
        return parsed["content"]
