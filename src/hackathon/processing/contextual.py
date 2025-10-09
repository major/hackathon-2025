"""Contextual Retrieval implementation using Anthropic's approach."""

import json

from openai import OpenAI
from sqlalchemy.orm import Session

from hackathon.config import get_settings
from hackathon.database.operations import create_contextual_chunk, get_node_ancestors
from hackathon.models.database import DocumentNode
from hackathon.models.schemas import ContextualChunkCreate
from hackathon.utils.logging import get_logger

logger = get_logger(__name__)


class ContextualRetriever:
    """Generate contextual summaries for document chunks."""

    def __init__(self) -> None:
        """Initialize the contextual retriever with OpenAI-compatible client."""
        self.settings = get_settings()

        # Initialize OpenAI client pointing to local granite4 server
        self.client = OpenAI(
            base_url=self.settings.llm_api_base, api_key=self.settings.llm_api_key
        )

    def build_context_from_ancestors(self, db: Session, node: DocumentNode) -> str:
        """
        Build context by walking up the document tree to ancestors.

        Args:
            db: Database session
            node: Current document node

        Returns:
            Context string from ancestors
        """
        ancestors = get_node_ancestors(db, node.id)

        # Build context from ancestors (root to immediate parent)
        context_parts = []
        for ancestor in reversed(ancestors):  # Reverse to go from root to parent
            if ancestor.text_content:
                context_parts.append(f"{ancestor.node_type}: {ancestor.text_content}")

        return " > ".join(context_parts) if context_parts else "Document"

    def generate_contextual_summary(
        self, db: Session, node: DocumentNode, document_context: str = ""
    ) -> str:
        """
        Generate a contextual summary for a chunk using the LLM.

        Args:
            db: Database session
            node: Document node to generate context for
            document_context: Additional document-level context

        Returns:
            Contextual summary string
        """
        # Get hierarchical context
        hierarchical_context = self.build_context_from_ancestors(db, node)

        # Construct prompt for contextual summary with JSON mode
        prompt = f"""<document>
{document_context}
</document>

Document hierarchy: {hierarchical_context}

Here is the chunk we want to situate within the whole document:
<chunk>
{node.text_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.

IMPORTANT: You must respond with ONLY valid JSON. Do not include any prefixes like "Context:", "Summary:", or any other text outside the JSON structure.

Respond with JSON in this exact format:
{{"context": "your succinct context here"}}

The value in the "context" field should be the contextual summary itself, without any prefixes or labels."""

        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
                response_format={"type": "json_object"},
            )

            content = (
                response.choices[0].message.content.strip() if response.choices else ""
            )

            if not content:
                logger.warning(
                    "Empty response from LLM, using hierarchical context as fallback"
                )
                return hierarchical_context

            # Parse JSON response
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                logger.exception(f"Failed to parse JSON response: {content[:200]}...")
                return hierarchical_context
            else:
                summary = result.get("context", "")

                if not summary:
                    logger.warning(
                        "Empty context in JSON response, using hierarchical context as fallback"
                    )
                    return hierarchical_context

                # Post-process: strip common prefixes that LLM might add
                prefixes_to_strip = ["Context:", "Summary:", "Context :", "Summary :"]
                for prefix in prefixes_to_strip:
                    summary = summary.removeprefix(prefix).strip()

                return summary
        except Exception:
            # Fallback to hierarchical context if LLM fails
            logger.exception(
                "LLM request failed. Using hierarchical context as fallback."
            )
            return hierarchical_context

    def create_contextual_chunk_for_node(
        self, db: Session, node: DocumentNode, document_context: str = ""
    ) -> None:
        """
        Create and store contextual chunk for a node.

        Args:
            db: Database session
            node: Document node
            document_context: Optional document-level context
        """
        if not node.text_content:
            return

        # Generate contextual summary
        summary = self.generate_contextual_summary(db, node, document_context)

        # Create contextualized text (summary + original)
        contextualized_text = f"{summary}\n\n{node.text_content}"

        # Store contextual chunk
        chunk_data = ContextualChunkCreate(
            node_id=node.id,
            original_text=node.text_content,
            contextual_summary=summary,
            contextualized_text=contextualized_text,
        )

        create_contextual_chunk(db, chunk_data)

    def batch_create_contextual_chunks(
        self,
        db: Session,
        nodes: list[DocumentNode],
        document_context: str = "",
        progress_callback=None,
    ) -> None:
        """
        Create contextual chunks for multiple nodes.

        Args:
            db: Database session
            nodes: List of document nodes
            document_context: Optional document-level context
            progress_callback: Optional callback for progress updates
        """
        total = len(nodes)

        for idx, node in enumerate(nodes):
            self.create_contextual_chunk_for_node(db, node, document_context)

            # Commit periodically
            if (idx + 1) % 10 == 0:
                db.commit()

            # Update progress
            if progress_callback:
                progress_callback(idx + 1, total)

        # Final commit
        db.commit()
