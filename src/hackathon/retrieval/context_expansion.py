"""Context expansion by walking up the document tree."""

import json

from sqlalchemy.orm import Session

from hackathon.database.operations import (
    get_all_leaf_nodes,
    get_node_ancestors,
    get_node_children,
)
from hackathon.models.database import DocumentNode
from hackathon.models.schemas import ExpandedContext


class ContextExpander:
    """Expand context by traversing the document tree."""

    def __init__(self, db: Session) -> None:
        """
        Initialize the context expander.

        Args:
            db: Database session
        """
        self.db = db

    def get_expanded_context(
        self,
        node: DocumentNode,
        include_ancestors: bool = True,
        include_children: bool = False,
    ) -> ExpandedContext:
        """
        Get expanded context for a node.

        Args:
            node: Document node to expand context for
            include_ancestors: Whether to include ancestor nodes
            include_children: Whether to include child nodes

        Returns:
            ExpandedContext with ancestors and children
        """
        # Build base context
        context = ExpandedContext(
            node_id=node.id,
            text_content=node.text_content or "",
            node_type=node.node_type,
            node_path=node.node_path,
        )

        # Add ancestors if requested
        if include_ancestors:
            ancestors = get_node_ancestors(self.db, node.id)
            context.parents = [
                ExpandedContext(
                    node_id=ancestor.id,
                    text_content=ancestor.text_content or "",
                    node_type=ancestor.node_type,
                    node_path=ancestor.node_path,
                )
                for ancestor in ancestors
            ]

        # Add children if requested
        if include_children:
            children = get_node_children(self.db, node.id)
            context.children = [
                ExpandedContext(
                    node_id=child.id,
                    text_content=child.text_content or "",
                    node_type=child.node_type,
                    node_path=child.node_path,
                )
                for child in children
            ]

        return context

    def build_context_text(
        self,
        node: DocumentNode,
        depth: int = 1,
        include_siblings: bool = False,
        exclude_current: bool = False,
    ) -> str:
        """
        Build expanded context text by walking up the tree.

        Args:
            node: Starting node
            depth: How many levels up to traverse (0 = just node, 1 = parent, 2 = grandparent, etc.)
            include_siblings: Whether to include sibling nodes
            exclude_current: If True, don't include the current node (useful when showing semantic block separately)

        Returns:
            Expanded context as formatted text
        """
        parts = []

        # Get ancestors up to specified depth
        ancestors = get_node_ancestors(self.db, node.id)[:depth]

        # Add ancestors from root down
        for ancestor in reversed(ancestors):
            if ancestor.text_content:
                parts.append(f"[{ancestor.node_type}] {ancestor.text_content}")

        # Add siblings if requested
        if include_siblings and node.parent_id:
            siblings = get_node_children(self.db, node.parent_id)
            for sibling in siblings:
                if sibling.id != node.id and sibling.text_content:
                    parts.append(
                        f"[sibling: {sibling.node_type}] {sibling.text_content}"
                    )

        # Add the node itself (unless excluded)
        if not exclude_current and node.text_content:
            parts.append(f"[current: {node.node_type}] {node.text_content}")

        return "\n\n".join(parts)

    def get_parent_section(self, node: DocumentNode) -> DocumentNode | None:
        """
        Get the parent section node.

        Args:
            node: Current node

        Returns:
            Parent section node if found, None otherwise
        """
        ancestors = get_node_ancestors(self.db, node.id)

        for ancestor in ancestors:
            if ancestor.node_type == "section":
                return ancestor

        return None

    def get_full_section_text(self, node: DocumentNode) -> str:
        """
        Get full text of the section containing this node.

        Args:
            node: Node within a section

        Returns:
            Full section text including all child nodes
        """
        # Find parent section
        section = self.get_parent_section(node)

        if not section:
            return node.text_content or ""

        # Get all children of the section
        children = get_node_children(self.db, section.id)

        # Build full text
        parts = []
        if section.text_content:
            parts.append(f"# {section.text_content}")

        for child in children:
            if child.text_content:
                parts.append(child.text_content)

        return "\n\n".join(parts)

    def get_semantic_block_text(self, node: DocumentNode) -> str:
        """
        Get the full semantic block (list, code block, table) if node is part of one.

        If the node is part of a semantic block that was split across multiple chunks,
        this returns the complete block by reassembling all related chunks.

        Args:
            node: Node that might be part of a semantic block

        Returns:
            Complete semantic block text, or just the node text if not in a block
        """
        # Check if node has semantic block metadata (legacy from old chunker)
        metadata = node.meta or {}
        semantic_block_str = metadata.get("semantic_block")

        if semantic_block_str:
            # Legacy semantic block metadata exists, use old logic
            return self._reassemble_legacy_semantic_block(node, semantic_block_str)

        # New Docling-based logic: check if this is a code block that might be split
        docling_types = metadata.get("docling_types", "")
        node_type = node.node_type

        # Only try to reassemble if this node contains code
        if node_type != "code" and "code" not in docling_types:
            return node.text_content or ""

        # Try to find adjacent chunks that are part of the same code block
        return self._reassemble_docling_code_block(node)

    def _reassemble_legacy_semantic_block(
        self, node: DocumentNode, semantic_block_str: str
    ) -> str:
        """Reassemble semantic blocks from legacy markdown_chunker metadata."""
        # Parse the semantic block metadata
        semantic_block = self._parse_semantic_block(semantic_block_str)
        if not semantic_block or not node.parent_id:
            return node.text_content or ""

        # Find related chunks
        block_chunks = self._find_related_legacy_chunks(node, semantic_block)
        if not block_chunks:
            return node.text_content or ""

        # Reassemble based on block type
        return self._join_chunks_by_type(block_chunks, semantic_block.get("type"))

    def _parse_semantic_block(self, semantic_block_str: str) -> dict | None:
        """Parse semantic block JSON string safely."""
        try:
            return (
                json.loads(semantic_block_str)
                if isinstance(semantic_block_str, str)
                else semantic_block_str
            )
        except (json.JSONDecodeError, TypeError):
            return None

    def _find_related_legacy_chunks(
        self, node: DocumentNode, semantic_block: dict
    ) -> list[DocumentNode]:
        """Find sibling chunks that are part of the same semantic block."""
        siblings = get_node_children(self.db, node.parent_id)
        block_chunks = []

        for sibling in siblings:
            sibling_block = self._parse_semantic_block(
                (sibling.meta or {}).get("semantic_block")
            )
            if self._is_same_semantic_block(sibling_block, semantic_block):
                block_chunks.append(sibling)

        block_chunks.sort(key=lambda n: n.node_path)
        return block_chunks

    def _is_same_semantic_block(
        self, sibling_block: dict | None, semantic_block: dict
    ) -> bool:
        """Check if two semantic blocks are the same."""
        if not sibling_block:
            return False
        return (
            sibling_block.get("type") == semantic_block.get("type")
            and sibling_block.get("start_line") == semantic_block.get("start_line")
            and sibling_block.get("end_line") == semantic_block.get("end_line")
        )

    def _join_chunks_by_type(
        self, chunks: list[DocumentNode], block_type: str | None
    ) -> str:
        """Join chunks based on block type."""
        texts = [chunk.text_content or "" for chunk in chunks]

        if block_type in ["code_block", "ordered_list", "unordered_list", "table"]:
            return "\n".join(texts)
        return "\n\n".join(texts)

    def _reassemble_docling_code_block(self, node: DocumentNode) -> str:
        """
        Reassemble code blocks that Docling split across multiple chunks.

        Args:
            node: The current code node

        Returns:
            Reassembled code block text or original node text
        """
        related_chunks = self._find_related_code_chunks(node)

        if len(related_chunks) <= 1:
            return node.text_content or ""

        related_chunks.sort(key=lambda n: n.node_path)
        return "\n".join(chunk.text_content or "" for chunk in related_chunks)

    def _find_related_code_chunks(self, node: DocumentNode) -> list[DocumentNode]:
        """Find code chunks that are part of the same code block."""
        metadata = node.meta or {}
        heading = metadata.get("headings", "")
        all_nodes = get_all_leaf_nodes(self.db, node.document_id)

        related_chunks = []
        for other_node in all_nodes:
            if self._is_related_code_chunk(other_node, heading, node.parent_id):
                related_chunks.append(other_node)

        return related_chunks

    def _is_related_code_chunk(
        self, other_node: DocumentNode, heading: str, parent_id: int | None
    ) -> bool:
        """Check if a node is part of the same code block."""
        other_meta = other_node.meta or {}
        other_heading = other_meta.get("headings", "")
        other_docling_types = other_meta.get("docling_types", "")

        return (
            other_heading == heading
            and other_node.parent_id == parent_id
            and (other_node.node_type == "code" or "code" in other_docling_types)
        )
