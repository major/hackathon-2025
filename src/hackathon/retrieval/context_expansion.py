"""Context expansion using neighbor-based retrieval (flat structure)."""

from sqlalchemy.orm import Session

from hackathon.database.node_ops import get_all_leaf_nodes, get_neighbors
from hackathon.models.database import DocumentNode


class ContextExpander:
    """Expand context using position-based neighbor retrieval."""

    def __init__(self, db: Session) -> None:
        """
        Initialize the context expander.

        Args:
            db: Database session
        """
        self.db = db

    def get_neighbor_context(
        self, node: DocumentNode, before: int = 1, after: int = 1
    ) -> str:
        """
        Get context by expanding to include N neighboring chunks before and after.

        This provides a simple, predictable expansion strategy based on document order.

        Args:
            node: The current node (search result)
            before: Number of chunks to include before the current node
            after: Number of chunks to include after the current node

        Returns:
            Formatted context text with neighbors clearly marked

        Example:
            get_neighbor_context(node, before=2, after=1) returns:
            [BEFORE-2] ...
            [BEFORE-1] ...
            [CURRENT] ...
            [AFTER-1] ...
        """
        parts = []

        # Get neighbors from database
        before_neighbors, after_neighbors = get_neighbors(self.db, node, before, after)

        # Add before neighbors (furthest to closest)
        for i, neighbor in enumerate(before_neighbors, start=1):
            distance = len(before_neighbors) - i + 1
            parts.append(self._format_neighbor_node(neighbor, f"BEFORE-{distance}"))

        # Add current node
        parts.append(self._format_node_content(node, "current"))

        # Add after neighbors (closest to furthest)
        for i, neighbor in enumerate(after_neighbors, start=1):
            parts.append(self._format_neighbor_node(neighbor, f"AFTER-{i}"))

        return "\n\n".join(parts)

    def get_semantic_block_text(self, node: DocumentNode) -> str:
        """
        Get the full semantic block text (for code blocks that might be split).

        For the flat HybridChunker structure, this just returns the node text
        since HybridChunker generally keeps semantic blocks intact.

        Args:
            node: Node that might be part of a semantic block

        Returns:
            Node text (semantic blocks are already intact with HybridChunker)
        """
        # With HybridChunker's intelligent splitting, semantic blocks are usually intact
        # If needed, we could use neighbors to reassemble based on heading context
        return node.text_content or ""

    def get_full_document_context(self, node: DocumentNode) -> str:
        """
        Get the entire document context (all chunks in order).

        Args:
            node: Any node in the document

        Returns:
            Complete document text with all nodes
        """
        # Get all leaf nodes for this document, ordered by position
        all_nodes = get_all_leaf_nodes(self.db, node.document_id)
        all_nodes.sort(key=lambda n: n.position if n.position is not None else 0)

        parts = []
        for doc_node in all_nodes:
            if doc_node.text_content:
                parts.append(self._format_node_content(doc_node, "document"))

        return "\n\n".join(parts)

    def get_parent_context(
        self, node: DocumentNode, levels: int = 1, _include_siblings: bool = False
    ) -> str:
        """
        Get context by expanding to N surrounding chunks (flat structure).

        Since we have a flat structure, "levels" means surrounding chunks.

        Args:
            node: The starting node
            levels: Number of surrounding chunks (0 = just node, N = N before + N after, -1 = full doc)
            _include_siblings: Ignored (no hierarchy, kept for API compatibility)

        Returns:
            Formatted context text
        """
        if levels == 0:
            return self._format_node_content(node, "current")

        if levels == -1:
            return self.get_full_document_context(node)

        # Use neighbor context for surrounding chunks
        return self.get_neighbor_context(node, before=levels, after=levels)

    def estimate_context_sizes(
        self, node: DocumentNode, max_levels: int = 5
    ) -> dict[str, int]:
        """
        Estimate character counts for different expansion levels.

        Args:
            node: The node to estimate expansions for
            max_levels: Maximum number of surrounding chunks to estimate

        Returns:
            Dictionary mapping expansion level to character count
        """
        sizes = {}

        # Node only
        sizes["node"] = len(node.text_content or "")

        # Surrounding chunks at different levels
        for level in range(1, max_levels + 1):
            context_text = self.get_parent_context(node, levels=level)
            sizes[f"level_{level}"] = len(context_text)

        # Full document
        full_doc = self.get_full_document_context(node)
        sizes["full_document"] = len(full_doc)

        return sizes

    def _format_node_content(self, node: DocumentNode, context_type: str) -> str:
        """Format a node's content with metadata header."""
        metadata = node.meta or {}
        header_parts = []

        # Map context types to display labels
        context_labels = {
            "current": "CURRENT",
            "before": "BEFORE",
            "after": "AFTER",
            "document": "DOCUMENT",
        }
        label = context_labels.get(context_type, context_type.upper())

        # Add node type and context indicator
        header_parts.append(f"[{label}: {node.node_type}]")

        # Add heading context if available
        if headings := metadata.get("headings"):
            header_parts.append(f"Section: {headings}")

        header = " - ".join(header_parts)

        if node.text_content:
            return f"{header}\n{node.text_content}"
        else:
            return f"{header}\n[No content]"

    def _format_semantic_block(self, node: DocumentNode, block_text: str) -> str:
        """Format a semantic block with metadata header."""
        metadata = node.meta or {}
        header_parts = []

        header_parts.append(f"[CURRENT: {node.node_type} - COMPLETE BLOCK]")

        if headings := metadata.get("headings"):
            header_parts.append(f"Section: {headings}")

        header = " - ".join(header_parts)
        return f"{header}\n{block_text}"

    def _format_neighbor_node(self, node: DocumentNode, label: str) -> str:
        """
        Format a neighbor node with a clear label.

        Args:
            node: The neighbor node
            label: Label like "BEFORE-2" or "AFTER-1"

        Returns:
            Formatted node text
        """
        metadata = node.meta or {}
        header_parts = []

        # Add position label and node type
        header_parts.append(f"[{label}: {node.node_type}]")

        # Add heading context if available
        if headings := metadata.get("headings"):
            header_parts.append(f"Section: {headings}")

        # Add position info
        if node.position is not None:
            header_parts.append(f"Position: {node.position}")

        header = " - ".join(header_parts)

        if node.text_content:
            return f"{header}\n{node.text_content}"
        else:
            return f"{header}\n[No content]"
