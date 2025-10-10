"""Tests for contextual summary generation using IBM Watsonx. 🤖"""

from unittest.mock import MagicMock, patch

import pytest

from hackathon.processing.contextual import (
    _build_contextual_prompt,
    _clean_llm_response,
    _fallback_summary,
    _is_valid_summary,
    generate_contextual_summary,
    generate_contextual_text,
    get_watsonx_credentials,
)


class TestWatsonxCredentials:
    """Test Watsonx credential retrieval."""

    def test_get_watsonx_credentials_success(self, mock_env_vars):
        """Test successful credential retrieval."""
        credentials = get_watsonx_credentials()

        assert credentials.api_key == "test-api-key"
        assert credentials.url == "https://us-south.ml.cloud.ibm.com"

    def test_get_watsonx_credentials_missing_api_key(self, monkeypatch):
        """Test error when API key is missing."""
        monkeypatch.setenv("WATSONX_API_KEY", "")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project")

        with pytest.raises(ValueError, match="WATSONX_API_KEY not set"):
            get_watsonx_credentials()

    def test_get_watsonx_credentials_missing_project_id(self, monkeypatch):
        """Test error when project ID is missing."""
        monkeypatch.setenv("WATSONX_API_KEY", "test-key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "")

        with pytest.raises(ValueError, match="WATSONX_PROJECT_ID not set"):
            get_watsonx_credentials()


class TestPromptBuilding:
    """Test contextual prompt construction."""

    def test_build_contextual_prompt_with_headings(self):
        """Test prompt building with heading context."""
        filename = "setup.md"
        headings = "Installation > Dependencies"
        chunk_text = "Run npm install to download packages."

        prompt = _build_contextual_prompt(filename, headings, chunk_text)

        assert filename in prompt
        assert headings in prompt
        assert chunk_text in prompt
        assert "1-2 sentence" in prompt.lower()
        # Should instruct NOT to add prefixes like "Context:" or "Summary:"
        assert "preamble" in prompt.lower() or "labels" in prompt.lower()

    def test_build_contextual_prompt_no_headings(self):
        """Test prompt building without heading context."""
        filename = "readme.md"
        headings = ""
        chunk_text = "This is the introduction."

        prompt = _build_contextual_prompt(filename, headings, chunk_text)

        assert filename in prompt
        assert "N/A" in prompt  # No headings
        assert chunk_text in prompt

    @pytest.mark.parametrize(
        "filename,headings,chunk_text",
        [
            ("config.md", "Database > PostgreSQL", "Set DB_HOST in .env file"),
            ("api.md", "Endpoints > Users", "GET /api/users returns user list"),
            ("test.md", "", "Simple test case"),
        ],
    )
    def test_build_contextual_prompt_various_inputs(
        self, filename, headings, chunk_text
    ):
        """Test prompt building with various inputs."""
        prompt = _build_contextual_prompt(filename, headings, chunk_text)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert chunk_text in prompt


class TestLLMResponseCleaning:
    """Test cleaning of LLM responses."""

    @pytest.mark.parametrize(
        "input_text,expected_output",
        [
            # Remove common prefixes
            ("Context: This explains setup", "This explains setup"),
            ("Summary: Installation guide", "Installation guide"),
            ("Contextual summary: Database config", "Database config"),
            ("This chunk is about testing", "is about testing"),
            ("Here is the summary", "the summary"),
            # Remove quotes
            ('"This is a summary"', "This is a summary"),
            ("'Another summary'", "Another summary"),
            # Multiple cleanings
            ('Summary: "Database setup"', "Database setup"),
            # Already clean
            ("Clean summary text", "Clean summary text"),
        ],
    )
    def test_clean_llm_response(self, input_text, expected_output):
        """Test cleaning various LLM response patterns."""
        result = _clean_llm_response(input_text)
        assert result == expected_output

    def test_clean_llm_response_whitespace(self):
        """Test that whitespace is properly trimmed."""
        result = _clean_llm_response("  \n  Summary: Text  \n  ")
        assert result == "Text"

    def test_clean_llm_response_multiple_prefixes(self):
        """Test removal of multiple common prefixes."""
        result = _clean_llm_response("The contextual summary is: Some text")
        assert result == "Some text"


class TestSummaryValidation:
    """Test summary validation logic."""

    @pytest.mark.parametrize(
        "summary,expected_valid",
        [
            ("This is a valid summary with enough text.", True),
            ("Short but valid summary.", True),
            ("Too short", False),
            ("", False),
            ("   ", False),
            ("Minimum ok", True),  # Exactly 10 chars
            ("Not enoug", False),  # 9 chars
        ],
    )
    def test_is_valid_summary(self, summary, expected_valid):
        """Test summary validation with various inputs."""
        result = _is_valid_summary(summary)
        assert result == expected_valid


class TestFallbackSummary:
    """Test fallback summary generation."""

    def test_fallback_summary(self):
        """Test fallback summary format."""
        filename = "test.md"
        result = _fallback_summary(filename)

        assert filename in result
        assert "Content from" in result


class TestContextualSummaryGeneration:
    """Test contextual summary generation with mocked Watsonx."""

    @patch("hackathon.processing.contextual.ModelInference")
    @patch("hackathon.processing.contextual.get_watsonx_credentials")
    def test_generate_contextual_summary_success(
        self, mock_credentials, mock_model_class, mock_env_vars
    ):
        """Test successful summary generation."""
        # Setup mocks
        mock_credentials.return_value = MagicMock(
            api_key="test-key", url="https://test.com"
        )
        mock_model = MagicMock()
        mock_model.generate_text.return_value = (
            "This section explains database configuration."
        )
        mock_model_class.return_value = mock_model

        chunk_text = "Set DB_HOST in your .env file"
        context = {"filename": "config.md", "headings": "Setup > Database"}

        result = generate_contextual_summary(chunk_text, context)

        assert result == "This section explains database configuration."
        mock_model.generate_text.assert_called_once()

    @patch("hackathon.processing.contextual.ModelInference")
    @patch("hackathon.processing.contextual.get_watsonx_credentials")
    def test_generate_contextual_summary_with_prefix_cleaning(
        self, mock_credentials, mock_model_class, mock_env_vars
    ):
        """Test that LLM response prefixes are cleaned."""
        mock_credentials.return_value = MagicMock(
            api_key="test-key", url="https://test.com"
        )
        mock_model = MagicMock()
        mock_model.generate_text.return_value = "Summary: This explains setup"
        mock_model_class.return_value = mock_model

        chunk_text = "Installation instructions"
        context = {"filename": "setup.md"}

        result = generate_contextual_summary(chunk_text, context)

        assert result == "This explains setup"
        assert not result.startswith("Summary:")

    @patch("hackathon.processing.contextual.logger")
    @patch("hackathon.processing.contextual.ModelInference")
    @patch("hackathon.processing.contextual.get_watsonx_credentials")
    def test_generate_contextual_summary_retry_on_short_response(
        self, mock_credentials, mock_model_class, mock_logger, mock_env_vars
    ):
        """Test retry logic when LLM returns too short response."""
        mock_credentials.return_value = MagicMock(
            api_key="test-key", url="https://test.com"
        )
        mock_model = MagicMock()
        # First call: too short, second call: valid
        mock_model.generate_text.side_effect = [
            "Too short",  # 9 chars - below 10 minimum
            "This is a valid contextual summary.",
        ]
        mock_model_class.return_value = mock_model

        chunk_text = "Test content"
        context = {"filename": "test.md", "headings": ""}

        result = generate_contextual_summary(chunk_text, context, max_retries=3)

        assert result == "This is a valid contextual summary."
        assert mock_model.generate_text.call_count == 2
        assert mock_logger.warning.called

    @patch("hackathon.processing.contextual.logger")
    @patch("hackathon.processing.contextual.ModelInference")
    @patch("hackathon.processing.contextual.get_watsonx_credentials")
    def test_generate_contextual_summary_api_error_fallback(
        self, mock_credentials, mock_model_class, mock_logger, mock_env_vars
    ):
        """Test fallback when API consistently fails."""
        mock_credentials.return_value = MagicMock(
            api_key="test-key", url="https://test.com"
        )
        mock_model = MagicMock()
        mock_model.generate_text.side_effect = Exception("API Error")
        mock_model_class.return_value = mock_model

        chunk_text = "Test content"
        context = {"filename": "test.md", "headings": ""}

        result = generate_contextual_summary(chunk_text, context, max_retries=2)

        # Should return fallback
        assert result == "Content from test.md"
        assert mock_model.generate_text.call_count == 2  # Retried
        assert mock_logger.warning.called or mock_logger.error.called

    @patch("hackathon.processing.contextual.logger")
    @patch("hackathon.processing.contextual.ModelInference")
    @patch("hackathon.processing.contextual.get_watsonx_credentials")
    def test_generate_contextual_summary_empty_response_fallback(
        self, mock_credentials, mock_model_class, mock_logger, mock_env_vars
    ):
        """Test fallback when LLM returns empty after all retries."""
        mock_credentials.return_value = MagicMock(
            api_key="test-key", url="https://test.com"
        )
        mock_model = MagicMock()
        mock_model.generate_text.return_value = ""  # Empty response
        mock_model_class.return_value = mock_model

        chunk_text = "Test content"
        context = {"filename": "test.md", "headings": ""}

        result = generate_contextual_summary(chunk_text, context, max_retries=2)

        # Should return fallback after retries
        assert result == "Content from test.md"
        assert mock_logger.warning.called

    @patch("hackathon.processing.contextual.ModelInference")
    @patch("hackathon.processing.contextual.get_watsonx_credentials")
    def test_generate_contextual_summary_uses_correct_model(
        self, mock_credentials, mock_model_class, mock_env_vars
    ):
        """Test that correct model and parameters are used."""
        mock_credentials.return_value = MagicMock(
            api_key="test-key", url="https://test.com"
        )
        mock_model = MagicMock()
        mock_model.generate_text.return_value = "Valid summary text here."
        mock_model_class.return_value = mock_model

        chunk_text = "Test"
        context = {"filename": "test.md"}

        generate_contextual_summary(chunk_text, context)

        # Verify ModelInference was called with correct parameters
        mock_model_class.assert_called_once()
        call_kwargs = mock_model_class.call_args[1]
        assert call_kwargs["model_id"] == "ibm/granite-4-h-small"
        assert call_kwargs["params"]["temperature"] == 0.1
        assert call_kwargs["params"]["max_new_tokens"] == 100


class TestContextualTextGeneration:
    """Test combining summary with chunk text."""

    @pytest.mark.parametrize(
        "chunk_text,summary,expected",
        [
            (
                "Run npm install",
                "This explains dependency installation.",
                "This explains dependency installation. Run npm install",
            ),
            (
                "Database configuration guide",
                "Setup instructions for PostgreSQL.",
                "Setup instructions for PostgreSQL. Database configuration guide",
            ),
            ("Short", "Context.", "Context. Short"),
        ],
    )
    def test_generate_contextual_text(self, chunk_text, summary, expected):
        """Test combining summary and chunk text."""
        result = generate_contextual_text(chunk_text, summary)
        assert result == expected

    def test_generate_contextual_text_spacing(self):
        """Test that spacing is correct between summary and text."""
        summary = "This is the summary"
        text = "This is the text"

        result = generate_contextual_text(text, summary)

        # Should have exactly one space between
        assert result == f"{summary} {text}"
        assert "  " not in result  # No double spaces
