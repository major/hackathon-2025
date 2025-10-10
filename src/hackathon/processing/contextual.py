"""Contextual summary generation using IBM Watsonx AI.

Implements Anthropic's Contextual Retrieval pattern using Watsonx LLMs
to generate situational context for each chunk.
"""

from typing import Any

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from hackathon.config import get_settings
from hackathon.utils.logging import get_logger

# 🔇 Logging configured centrally in hackathon/__init__.py
logger = get_logger(__name__)


def get_watsonx_credentials() -> Credentials:
    """Get IBM Watsonx credentials from settings."""
    settings = get_settings()
    if not settings.watsonx_api_key:
        msg = "WATSONX_API_KEY not set in environment. Get your key at: https://cloud.ibm.com/iam/apikeys"
        raise ValueError(msg)
    if not settings.watsonx_project_id:
        msg = "WATSONX_PROJECT_ID not set in environment"
        raise ValueError(msg)

    return Credentials(
        url=settings.watsonx_url,
        api_key=settings.watsonx_api_key,
    )


def generate_contextual_summary(
    chunk_text: str,
    document_context: dict[str, Any],
    max_retries: int = 3,
) -> str:
    """
    Generate a contextual summary for a chunk using IBM Watsonx AI.

    Following Anthropic's Contextual Retrieval pattern, this generates a brief
    contextual description that situates the chunk within the broader document.

    Args:
        chunk_text: The text content of the chunk
        document_context: Document metadata (filename, headings, etc.)
        max_retries: Maximum number of retries on API errors

    Returns:
        Contextual summary (1-2 sentences)

    Example:
        >>> context = {"filename": "setup.md", "headings": "Installation > Dependencies"}
        >>> generate_contextual_summary("Run npm install...", context)
        "This section explains how to install project dependencies using npm."
    """
    settings = get_settings()
    credentials = get_watsonx_credentials()
    filename = document_context.get("filename", "unknown")
    headings = document_context.get("headings", "")

    prompt = _build_contextual_prompt(filename, headings, chunk_text)
    model = _create_watsonx_model(settings, credentials)

    for attempt in range(max_retries):
        try:
            response = model.generate_text(prompt=prompt)
            summary = _clean_llm_response(response)

            if _is_valid_summary(summary):
                return summary

            logger.warning(
                f"Empty or too short contextual summary (attempt {attempt + 1}/{max_retries})"
            )

        except Exception as e:
            logger.warning(
                f"Watsonx API error (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt == max_retries - 1:
                logger.error(
                    f"Failed to generate contextual summary after {max_retries} attempts"
                )
                return _fallback_summary(filename)

    return _fallback_summary(filename)


def _build_contextual_prompt(filename: str, headings: str, chunk_text: str) -> str:
    """Build the prompt for contextual summary generation."""
    return f"""Here is a chunk from a document titled "{filename}".

Document context:
- Heading hierarchy: {headings if headings else "N/A"}

Chunk content:
{chunk_text}

Please provide a brief, 1-2 sentence contextual summary that explains what this chunk is about and how it relates to the broader document. Focus on the topic and purpose, not the specific details.

Your response should be ONLY the contextual summary, with no preamble or labels like "Context:" or "Summary:".
"""


def _create_watsonx_model(settings, credentials):
    """Create a Watsonx model instance for text generation."""
    return ModelInference(
        model_id=settings.watsonx_llm_model,
        credentials=credentials,
        project_id=settings.watsonx_project_id,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 100,
            "temperature": 0.1,  # Low temperature for consistent summaries
        },
    )


def _clean_llm_response(response: str) -> str:
    """Clean up common artifacts from LLM responses."""
    summary = response.strip()

    # Remove common prefixes
    prefixes = [
        "Context:",
        "Summary:",
        "Contextual summary:",
        "This chunk",
        "Here is",
        "The contextual summary is:",
    ]
    for prefix in prefixes:
        if summary.startswith(prefix):
            summary = summary[len(prefix) :].strip()

    # Remove quotes
    if summary.startswith('"') and summary.endswith('"'):
        summary = summary[1:-1]
    if summary.startswith("'") and summary.endswith("'"):
        summary = summary[1:-1]

    return summary


def _is_valid_summary(summary: str) -> bool:
    """Check if the summary is valid (not empty, reasonable length)."""
    return bool(summary and len(summary) >= 10)


def _fallback_summary(filename: str) -> str:
    """Generate a fallback summary when LLM fails."""
    return f"Content from {filename}"


def generate_contextual_text(
    chunk_text: str,
    contextual_summary: str,
) -> str:
    """
    Combine contextual summary with chunk text for indexing.

    This creates the enriched text that will be indexed in BM25,
    following the Anthropic cookbook pattern.

    Args:
        chunk_text: Original chunk text
        contextual_summary: LLM-generated contextual summary

    Returns:
        Combined text for indexing (summary + original text)

    Example:
        >>> summary = "This section explains database configuration."
        >>> text = "Set DB_HOST=localhost in .env file"
        >>> generate_contextual_text(text, summary)
        "This section explains database configuration. Set DB_HOST=localhost in .env file"
    """
    return f"{contextual_summary} {chunk_text}"
