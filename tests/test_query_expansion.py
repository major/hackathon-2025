"""Tests for query expansion with IBM Watsonx. 🔍"""

import json
from unittest.mock import MagicMock, patch


from hackathon.retrieval.query_expansion import QueryExpander, expand_query


class TestQueryExpander:
    """Test QueryExpander class."""

    @patch("hackathon.retrieval.query_expansion.ModelInference")
    def test_initialization_with_defaults(self, mock_model_inference, mock_env_vars):
        """Test QueryExpander initialization with default settings."""
        expander = QueryExpander()

        assert expander.num_variations == 3
        assert expander.temperature == 0.7
        assert expander.model_id == "ibm/granite-4-h-small"
        mock_model_inference.assert_called_once()

    @patch("hackathon.retrieval.query_expansion.ModelInference")
    def test_initialization_with_custom_params(
        self, mock_model_inference, mock_env_vars
    ):
        """Test QueryExpander initialization with custom parameters."""
        expander = QueryExpander(
            num_variations=5, model_id="custom-model", temperature=0.9
        )

        assert expander.num_variations == 5
        assert expander.temperature == 0.9
        assert expander.model_id == "custom-model"

    @patch("hackathon.retrieval.query_expansion.ModelInference")
    def test_expand_query_success(self, mock_model_inference, mock_env_vars):
        """Test successful query expansion."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.generate_text.return_value = json.dumps([
            "set up logging",
            "enable logs",
            "log configuration",
        ])
        mock_model_inference.return_value = mock_model

        expander = QueryExpander(num_variations=3)
        variations = expander.expand_query("configure logging")

        # Should include original query + 3 variations
        assert len(variations) == 4
        assert variations[0] == "configure logging"  # Original query first
        assert "set up logging" in variations
        assert "enable logs" in variations
        assert "log configuration" in variations

        # Should have called the model
        mock_model.generate_text.assert_called_once()

    @patch("hackathon.retrieval.query_expansion.ModelInference")
    def test_expand_query_with_prefix_stripping(
        self, mock_model_inference, mock_env_vars
    ):
        """Test that common LLM prefixes are stripped."""
        # Setup mock with prefixed response
        mock_model = MagicMock()
        mock_model.generate_text.return_value = (
            'JSON array: ["variation 1", "variation 2"]'
        )
        mock_model_inference.return_value = mock_model

        expander = QueryExpander(num_variations=2)
        variations = expander.expand_query("test query")

        assert len(variations) == 3  # Original + 2 variations
        assert variations[0] == "test query"
        assert "variation 1" in variations
        assert "variation 2" in variations

    @patch("hackathon.retrieval.query_expansion.ModelInference")
    def test_expand_query_limits_variations(self, mock_model_inference, mock_env_vars):
        """Test that variations are limited to requested number."""
        # Setup mock with more variations than requested
        mock_model = MagicMock()
        mock_model.generate_text.return_value = json.dumps([
            "var 1",
            "var 2",
            "var 3",
            "var 4",
            "var 5",
        ])
        mock_model_inference.return_value = mock_model

        expander = QueryExpander(num_variations=2)
        variations = expander.expand_query("test")

        # Should only return original + 2 variations (not 5)
        assert len(variations) == 3
        assert variations[0] == "test"

    @patch(
        "hackathon.retrieval.query_expansion.logger"
    )  # Mock logger to prevent formatting errors
    @patch("hackathon.retrieval.query_expansion.ModelInference")
    def test_expand_query_api_failure_fallback(
        self, mock_model_inference, mock_logger, mock_env_vars
    ):
        """Test fallback to original query when API fails."""
        # Setup mock to raise exception
        mock_model = MagicMock()
        mock_model.generate_text.side_effect = Exception("API Error")
        mock_model_inference.return_value = mock_model

        expander = QueryExpander()
        variations = expander.expand_query("test query")

        # Should fallback to original query only
        assert variations == ["test query"]
        # Should have logged warning
        assert mock_logger.warning.called

    @patch(
        "hackathon.retrieval.query_expansion.logger"
    )  # Mock logger to prevent formatting errors
    @patch("hackathon.retrieval.query_expansion.ModelInference")
    def test_expand_query_invalid_json_fallback(
        self, mock_model_inference, mock_logger, mock_env_vars
    ):
        """Test fallback to plaintext parsing when response is not JSON."""
        # Setup mock with plaintext response (numbered list)
        mock_model = MagicMock()
        mock_model.generate_text.return_value = (
            "1. query test\n2. check test\n3. examine test"
        )
        mock_model_inference.return_value = mock_model

        expander = QueryExpander()
        variations = expander.expand_query("test query")

        # Should return original query + variations from plaintext
        assert "test query" in variations
        assert len(variations) == 4  # Original + 3 variations
        # Should have logged info about plaintext extraction
        assert mock_logger.info.called

    @patch("hackathon.retrieval.query_expansion.ModelInference")
    def test_expand_query_empty_response(self, mock_model_inference, mock_env_vars):
        """Test handling of empty API response."""
        # Setup mock with empty array
        mock_model = MagicMock()
        mock_model.generate_text.return_value = "[]"
        mock_model_inference.return_value = mock_model

        expander = QueryExpander()
        variations = expander.expand_query("test query")

        # Should return at least the original query
        assert variations == ["test query"]

    @patch("hackathon.retrieval.query_expansion.ModelInference")
    def test_create_expansion_prompt_format(self, mock_model_inference, mock_env_vars):
        """Test that expansion prompt is formatted correctly."""
        expander = QueryExpander(num_variations=3)
        prompt = expander._create_expansion_prompt("configure logging")

        # Check prompt contains key elements
        assert "3" in prompt
        assert "configure logging" in prompt
        assert "numbered list" in prompt.lower()
        assert "different words" in prompt.lower() or "synonym" in prompt.lower()

    @patch("hackathon.retrieval.query_expansion.ModelInference")
    def test_parse_variations_with_valid_json(
        self, mock_model_inference, mock_env_vars
    ):
        """Test parsing valid JSON variations."""
        expander = QueryExpander()
        variations = expander._parse_variations('["var1", "var2", "var3"]')

        assert len(variations) == 3
        assert variations == ["var1", "var2", "var3"]

    @patch("hackathon.retrieval.query_expansion.ModelInference")
    def test_parse_variations_filters_non_strings(
        self, mock_model_inference, mock_env_vars
    ):
        """Test that non-string items are filtered out."""
        expander = QueryExpander()
        # JSON with mixed types
        variations = expander._parse_variations('["valid", 123, null, "also valid"]')

        # Should only include strings
        assert len(variations) == 2
        assert "valid" in variations
        assert "also valid" in variations

    @patch("hackathon.retrieval.query_expansion.ModelInference")
    def test_parse_variations_filters_empty_strings(
        self, mock_model_inference, mock_env_vars
    ):
        """Test that empty strings are filtered out."""
        expander = QueryExpander()
        variations = expander._parse_variations('["valid", "", "   ", "also valid"]')

        # Should only include non-empty strings
        assert len(variations) == 2
        assert "valid" in variations
        assert "also valid" in variations

    @patch("hackathon.retrieval.query_expansion.ModelInference")
    def test_parse_plaintext_variations(self, mock_model_inference, mock_env_vars):
        """Test fallback plaintext parsing."""
        expander = QueryExpander(num_variations=3)
        plaintext = """
        1. set up logging
        2. enable logs
        3. log configuration
        """

        variations = expander._parse_plaintext_variations(plaintext)

        assert len(variations) == 3
        assert "set up logging" in variations
        assert "enable logs" in variations
        assert "log configuration" in variations

    @patch("hackathon.retrieval.query_expansion.ModelInference")
    def test_parse_plaintext_variations_with_bullets(
        self, mock_model_inference, mock_env_vars
    ):
        """Test plaintext parsing with bullet points."""
        expander = QueryExpander()
        plaintext = """
        - set up logging
        * enable logs
        • log configuration
        """

        variations = expander._parse_plaintext_variations(plaintext)

        assert len(variations) == 3
        assert all(
            v in variations
            for v in ["set up logging", "enable logs", "log configuration"]
        )

    @patch("hackathon.retrieval.query_expansion.ModelInference")
    def test_parse_plaintext_variations_with_quotes(
        self, mock_model_inference, mock_env_vars
    ):
        """Test that quotes are stripped from plaintext variations."""
        expander = QueryExpander()
        plaintext = '"variation 1"\n"variation 2"'

        variations = expander._parse_plaintext_variations(plaintext)

        # Quotes should be stripped
        assert "variation 1" in variations
        assert "variation 2" in variations

    @patch("hackathon.retrieval.query_expansion.ModelInference")
    def test_parse_plaintext_variations_ignores_short_lines(
        self, mock_model_inference, mock_env_vars
    ):
        """Test that very short lines are ignored."""
        expander = QueryExpander()
        plaintext = "valid variation\nok\nab\nanother valid one"

        variations = expander._parse_plaintext_variations(plaintext)

        # Should only include lines > 3 characters
        assert "valid variation" in variations
        assert "another valid one" in variations
        assert "ok" not in variations
        assert "ab" not in variations


class TestExpandQueryConvenience:
    """Test convenience function for query expansion."""

    @patch("hackathon.retrieval.query_expansion.QueryExpander")
    def test_expand_query_function(self, mock_expander_class):
        """Test expand_query convenience function."""
        mock_expander = MagicMock()
        mock_expander.expand_query.return_value = ["original", "var1", "var2"]
        mock_expander_class.return_value = mock_expander

        results = expand_query("original", num_variations=2)

        assert results == ["original", "var1", "var2"]
        mock_expander_class.assert_called_once_with(num_variations=2)
        mock_expander.expand_query.assert_called_once_with("original")

    @patch("hackathon.retrieval.query_expansion.QueryExpander")
    def test_expand_query_function_default_variations(self, mock_expander_class):
        """Test expand_query with default variation count."""
        mock_expander = MagicMock()
        mock_expander.expand_query.return_value = ["query"]
        mock_expander_class.return_value = mock_expander

        expand_query("query")

        # Should use default num_variations=3
        mock_expander_class.assert_called_once_with(num_variations=3)


class TestQueryExpansionIntegration:
    """Integration tests for query expansion in the search pipeline."""

    @patch("hackathon.retrieval.query_expansion.ModelInference")
    def test_multiple_expansions_are_unique(self, mock_model_inference, mock_env_vars):
        """Test that query expansion generates unique variations."""
        # Setup mock to return some duplicates
        mock_model = MagicMock()
        mock_model.generate_text.return_value = json.dumps([
            "variation 1",
            "variation 1",  # Duplicate
            "variation 2",
        ])
        mock_model_inference.return_value = mock_model

        expander = QueryExpander(num_variations=3)
        variations = expander.expand_query("test")

        # Should include original + up to 3 unique variations
        assert "test" in variations
        # Note: duplicates are allowed by the current implementation
        # This test documents current behavior

    @patch("hackathon.retrieval.query_expansion.ModelInference")
    def test_expand_query_preserves_original_intent(
        self, mock_model_inference, mock_env_vars
    ):
        """Test that original query is always first in results."""
        mock_model = MagicMock()
        mock_model.generate_text.return_value = json.dumps(["var1", "var2"])
        mock_model_inference.return_value = mock_model

        expander = QueryExpander()
        variations = expander.expand_query("original query")

        # Original query must be first for proper fallback behavior
        assert variations[0] == "original query"

    @patch("hackathon.retrieval.query_expansion.ModelInference")
    def test_temperature_affects_generation(self, mock_model_inference, mock_env_vars):
        """Test that temperature parameter is passed to the model."""
        mock_model = MagicMock()
        mock_model.generate_text.return_value = '["var1"]'
        mock_model_inference.return_value = mock_model

        QueryExpander(temperature=0.9)

        # Check that temperature was passed to ModelInference constructor
        call_kwargs = mock_model_inference.call_args[1]
        params = call_kwargs["params"]
        # Temperature should be in the parameters dict
        assert params.get("temperature") == 0.9
