"""Query expansion using IBM Watsonx for generating semantic variations.

This module implements query expansion to improve recall by generating
alternative phrasings, synonyms, and related terms for user queries.
"""

import json

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from hackathon.config import get_settings
from hackathon.utils.logging import get_logger

logger = get_logger(__name__)


class QueryExpander:
    """Expands queries using IBM Watsonx LLM to generate semantic variations.

    This class uses IBM Watsonx Granite to generate alternative phrasings,
    synonyms, and related terms for a given query. The expanded queries are
    then used for multi-query retrieval with RRF fusion.

    Example:
        >>> expander = QueryExpander()
        >>> variations = expander.expand_query("configure logging")
        >>> print(variations)
        ['configure logging', 'set up logs', 'enable logging', 'log configuration']
    """

    def __init__(
        self,
        num_variations: int = 3,
        model_id: str | None = None,
        temperature: float = 0.7,
    ) -> None:
        """Initialize the query expander.

        Args:
            num_variations: Number of query variations to generate (default: 3)
            model_id: Watsonx model ID (defaults to config.watsonx_llm_model)
            temperature: LLM temperature for variation diversity (default: 0.7)
        """
        self.num_variations = num_variations
        self.temperature = temperature

        settings = get_settings()
        self.model_id = model_id or settings.watsonx_llm_model

        # Initialize Watsonx model (using same pattern as contextual.py)
        credentials = Credentials(
            api_key=settings.watsonx_api_key,
            url=settings.watsonx_url,
        )

        # Create model with parameters (not passed to generate_text)
        self.model = ModelInference(
            model_id=self.model_id,
            credentials=credentials,
            project_id=settings.watsonx_project_id,
            params={
                "decoding_method": "greedy",
                "max_new_tokens": 200,
                "temperature": self.temperature,
            },
        )

        logger.info(
            f"Initialized QueryExpander with model={self.model_id}, "
            f"num_variations={num_variations}, temperature={temperature}"
        )

    def expand_query(self, query: str) -> list[str]:
        """Expand a query into semantic variations.

        Generates alternative phrasings, synonyms, and related terms for the
        input query. The original query is always included as the first item.

        Args:
            query: The original user query to expand

        Returns:
            List of query variations (original query + generated variations)

        Example:
            >>> expander.expand_query("how to install dependencies")
            ['how to install dependencies', 'installing required packages',
             'setup project dependencies', 'add package dependencies']
        """
        logger.info(f"Expanding query: '{query}'")

        # Create prompt for query expansion
        prompt = self._create_expansion_prompt(query)

        try:
            # Parameters are already set in ModelInference constructor
            response = self.model.generate_text(prompt=prompt)

            # Check if response is empty
            if not response or not response.strip():
                logger.warning(
                    "LLM returned empty response. Using original query only."
                )
                return [query]

            variations = self._parse_variations(response)

            # Always include original query as first item
            all_queries = [query] + variations

            logger.info(f"Generated {len(variations)} variations: {variations}")
            return all_queries

        except Exception as e:
            logger.warning(
                f"Query expansion failed: {e}. Using original query only.",
                exc_info=True,
            )
            # Fallback to original query if expansion fails
            return [query]

    def _create_expansion_prompt(self, query: str) -> str:
        """Create the LLM prompt for query expansion.

        Args:
            query: The original user query

        Returns:
            Formatted prompt string for the LLM
        """
        return f"""Task: Rephrase the following search query {self.num_variations} times using different words.

Original query: "{query}"

Requirements:
- Each variation must use different wording while keeping the same meaning
- Use synonyms and alternative phrasings
- Keep similar length and style to the original
- Make them suitable for keyword search

Format your response as a numbered list with each variation on its own line.

Example for "configure logging":
1. set up logging system
2. enable log files
3. adjust logging settings

Now write {self.num_variations} variations for "{query}" (do NOT just write numbers - write actual alternative phrasings):
"""

    def _parse_variations(self, response: str) -> list[str]:
        """Parse LLM response to extract query variations.

        Args:
            response: Raw LLM response text

        Returns:
            List of parsed query variations
        """
        # Clean up the response
        response = response.strip()

        # Remove common prefixes that LLMs sometimes add
        response = response.removeprefix("JSON array:")
        response = response.removeprefix("Here are the variations:")
        response = response.removeprefix("Here are")
        response = response.strip()

        try:
            # Parse JSON array
            variations = json.loads(response)

            if not isinstance(variations, list):
                logger.warning(
                    f"Expected list, got {type(variations)}. Using empty list."
                )
                return []

            # Filter out non-string items and empty strings
            variations = [
                str(v).strip() for v in variations if isinstance(v, str) and v.strip()
            ]

            # Limit to requested number
            variations = variations[: self.num_variations]

            return variations

        except json.JSONDecodeError:
            # LLM returned plaintext format - try plaintext parser
            plaintext_variations = self._parse_plaintext_variations(response)
            if plaintext_variations:
                logger.info(
                    f"Extracted {len(plaintext_variations)} variations from plaintext response"
                )
            else:
                logger.warning(
                    f"Failed to extract variations from response (first 200 chars): '{response[:200]}'"
                )
            return plaintext_variations

    def _parse_plaintext_variations(self, response: str) -> list[str]:
        """Fallback parser for non-JSON responses.

        Attempts to extract variations from plain text by splitting on newlines
        and removing numbering/bullets.

        Args:
            response: Plain text response

        Returns:
            List of extracted variations
        """
        lines = response.split("\n")
        variations = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove common numbering/bullet patterns
            line = line.removeprefix("-").strip()
            line = line.removeprefix("*").strip()
            line = line.removeprefix("•").strip()

            # Remove numbered list patterns (1., 2., etc.)
            if line and line[0].isdigit():
                # Find the first non-digit, non-dot, non-space character
                for i, char in enumerate(line):
                    if char not in "0123456789. ":
                        line = line[i:].strip()
                        break

            # Remove quotes if present
            line = line.strip('"').strip("'")

            if line and len(line) > 3:  # Ignore very short lines
                variations.append(line)

        return variations[: self.num_variations]


def expand_query(query: str, num_variations: int = 3) -> list[str]:
    """Convenience function for query expansion.

    Args:
        query: The query to expand
        num_variations: Number of variations to generate

    Returns:
        List of query variations (original + generated)
    """
    expander = QueryExpander(num_variations=num_variations)
    return expander.expand_query(query)
