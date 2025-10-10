"""Tests for Watsonx reranking. 🎯"""

from unittest.mock import MagicMock, patch

import pytest

from hackathon.models.schemas import SearchResult
from hackathon.retrieval.reranker import (
    MAX_DOCUMENT_CHARS,
    WatsonxReranker,
    _map_reranked_results,
    _truncate_documents,
    _validate_query_length,
    rerank_results,
)


class TestTruncateDocuments:
    """Test document truncation for token limits."""

    def test_truncate_documents_under_limit(self):
        """Test documents under the character limit are not truncated."""
        docs = ["Short document", "Another short one", "Brief text"]

        truncated, count = _truncate_documents(docs)

        assert truncated == docs
        assert count == 0

    def test_truncate_documents_over_limit(self):
        """Test documents over limit are truncated."""
        long_doc = "x" * (MAX_DOCUMENT_CHARS + 100)
        docs = ["Short", long_doc, "Another short"]

        truncated, count = _truncate_documents(docs)

        assert len(truncated) == 3
        assert truncated[0] == "Short"
        assert truncated[2] == "Another short"
        # Long doc should be truncated
        assert len(truncated[1]) == MAX_DOCUMENT_CHARS + 3  # +3 for "..."
        assert truncated[1].endswith("...")
        assert count == 1

    def test_truncate_documents_all_long(self):
        """Test all documents exceeding limit."""
        long_docs = ["x" * (MAX_DOCUMENT_CHARS + 50) for _ in range(3)]

        truncated, count = _truncate_documents(long_docs)

        assert all(len(doc) == MAX_DOCUMENT_CHARS + 3 for doc in truncated)
        assert all(doc.endswith("...") for doc in truncated)
        assert count == 3

    def test_truncate_documents_empty_list(self):
        """Test truncating empty list."""
        truncated, count = _truncate_documents([])

        assert truncated == []
        assert count == 0

    @pytest.mark.parametrize(
        "doc_length,expected_truncated",
        [
            (MAX_DOCUMENT_CHARS - 1, False),
            (MAX_DOCUMENT_CHARS, False),
            (MAX_DOCUMENT_CHARS + 1, True),
            (MAX_DOCUMENT_CHARS * 2, True),
        ],
    )
    def test_truncate_documents_boundary(self, doc_length, expected_truncated):
        """Test truncation at boundary conditions."""
        doc = "x" * doc_length
        truncated, count = _truncate_documents([doc])

        if expected_truncated:
            assert count == 1
            assert truncated[0].endswith("...")
        else:
            assert count == 0
            assert truncated[0] == doc


class TestValidateQueryLength:
    """Test query length validation and warnings."""

    @pytest.mark.parametrize(
        "query,expected_words",
        [
            ("single", 1),
            ("two words", 2),
            ("three word query", 3),
            ("configure logging in application", 4),
        ],
    )
    @patch("hackathon.retrieval.reranker.logger")
    def test_validate_query_length_word_count(self, mock_logger, query, expected_words):
        """Test query length validation with various word counts."""
        _validate_query_length(query)

        if expected_words <= 2:
            # Should warn for short queries
            assert mock_logger.warning.called
        else:
            # Should not warn for longer queries
            assert not mock_logger.warning.called

    @patch("hackathon.retrieval.reranker.logger")
    def test_validate_query_length_empty(self, mock_logger):
        """Test validation with empty query."""
        _validate_query_length("")
        assert mock_logger.warning.called

    @patch("hackathon.retrieval.reranker.logger")
    def test_validate_query_length_whitespace(self, mock_logger):
        """Test validation with whitespace-only query."""
        _validate_query_length("   \n  ")
        assert mock_logger.warning.called


class TestMapRerankedResults:
    """Test mapping reranked items to SearchResult objects."""

    def test_map_reranked_results_basic(self, sample_search_results):
        """Test basic reranking result mapping."""
        reranked_items = [
            {"index": 2, "score": 0.95},
            {"index": 0, "score": 0.87},
            {"index": 1, "score": 0.72},
        ]

        results = _map_reranked_results(reranked_items, sample_search_results)

        assert len(results) == 3
        # Should be in reranked order
        assert results[0].node_id == sample_search_results[2].node_id
        assert results[1].node_id == sample_search_results[0].node_id
        assert results[2].node_id == sample_search_results[1].node_id

        # Scores should be updated
        assert results[0].score == 0.95
        assert results[1].score == 0.87
        assert results[2].score == 0.72

    def test_map_reranked_results_metadata(self, sample_search_results):
        """Test that original scores are preserved in metadata."""
        reranked_items = [
            {"index": 1, "score": 0.99},
        ]

        results = _map_reranked_results(reranked_items, sample_search_results)

        assert results[0].metadata is not None
        assert "original_score" in results[0].metadata
        assert "bm25_rank" in results[0].metadata
        assert "rerank_score" in results[0].metadata
        assert results[0].metadata["bm25_rank"] == 2  # index + 1

    def test_map_reranked_results_invalid_index(self, sample_search_results):
        """Test handling of invalid indices."""
        reranked_items = [
            {"index": 0, "score": 0.95},
            {"index": 999, "score": 0.80},  # Invalid index
        ]

        results = _map_reranked_results(reranked_items, sample_search_results)

        # Should only include valid result
        assert len(results) == 1
        assert results[0].score == 0.95

    def test_map_reranked_results_empty(self):
        """Test mapping with empty results."""
        results = _map_reranked_results([], [])
        assert results == []


@patch("hackathon.retrieval.reranker.logger")  # Patch logger for entire class
class TestWatsonxReranker:
    """Test WatsonxReranker class."""

    @patch("hackathon.retrieval.reranker.Rerank")
    def test_reranker_initialization_with_env(
        self, mock_rerank_class, mock_logger, mock_env_vars
    ):
        """Test reranker initialization with environment variables."""
        reranker = WatsonxReranker()

        assert reranker.api_key == "test-api-key"
        assert reranker.project_id == "test-project-id"
        assert reranker.model == "cross-encoder/ms-marco-minilm-l-12-v2"
        mock_rerank_class.assert_called_once()

    @patch("hackathon.retrieval.reranker.Rerank")
    def test_reranker_initialization_with_params(self, mock_rerank_class, mock_logger):
        """Test reranker initialization with explicit parameters."""
        reranker = WatsonxReranker(
            api_key="custom-key", project_id="custom-project", model="custom-model"
        )

        assert reranker.api_key == "custom-key"
        assert reranker.project_id == "custom-project"
        assert reranker.model == "custom-model"

    def test_reranker_initialization_missing_api_key(self, mock_logger, monkeypatch):
        """Test error when API key is missing."""
        monkeypatch.setenv("WATSONX_API_KEY", "")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project")

        with pytest.raises(ValueError, match="API key not found"):
            WatsonxReranker()

    def test_reranker_initialization_missing_project_id(self, mock_logger, monkeypatch):
        """Test error when project ID is missing."""
        monkeypatch.setenv("WATSONX_API_KEY", "test-key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "")

        with pytest.raises(ValueError, match="project ID not found"):
            WatsonxReranker()

    @patch("hackathon.retrieval.reranker.Rerank")
    def test_rerank_success(
        self, mock_rerank_class, mock_logger, mock_env_vars, sample_search_results
    ):
        """Test successful reranking."""
        # Setup mock
        mock_reranker = MagicMock()
        mock_reranker.generate.return_value = {
            "results": [
                {"index": 2, "score": 0.95},
                {"index": 0, "score": 0.87},
                {"index": 1, "score": 0.72},
            ]
        }
        mock_rerank_class.return_value = mock_reranker

        reranker = WatsonxReranker()
        results = reranker.rerank("test query", sample_search_results, top_n=3)

        assert len(results) == 3
        # Should be reranked by score
        assert results[0].score == 0.95
        assert results[1].score == 0.87
        assert results[2].score == 0.72
        mock_reranker.generate.assert_called_once()

    @patch("hackathon.retrieval.reranker.Rerank")
    def test_rerank_top_n_limit(
        self, mock_rerank_class, mock_logger, mock_env_vars, sample_search_results
    ):
        """Test that top_n limits results."""
        mock_reranker = MagicMock()
        mock_reranker.generate.return_value = {
            "results": [
                {"index": 2, "score": 0.95},
                {"index": 0, "score": 0.87},
                {"index": 1, "score": 0.72},
            ]
        }
        mock_rerank_class.return_value = mock_reranker

        reranker = WatsonxReranker()
        results = reranker.rerank("test query", sample_search_results, top_n=2)

        # Should only return top 2
        assert len(results) == 2
        assert results[0].score == 0.95
        assert results[1].score == 0.87

    @patch("hackathon.retrieval.reranker.Rerank")
    def test_rerank_empty_results(self, mock_rerank_class, mock_logger, mock_env_vars):
        """Test reranking with empty results."""
        reranker = WatsonxReranker()
        results = reranker.rerank("test query", [], top_n=5)

        assert results == []

    @patch("hackathon.retrieval.reranker.Rerank")
    def test_rerank_api_error_fallback(
        self, mock_rerank_class, mock_logger, mock_env_vars, sample_search_results
    ):
        """Test fallback to original ranking when API fails."""
        mock_reranker = MagicMock()
        mock_reranker.generate.side_effect = Exception("API Error")
        mock_rerank_class.return_value = mock_reranker

        reranker = WatsonxReranker()
        results = reranker.rerank("test query", sample_search_results, top_n=2)

        # Should return original top_n results
        assert len(results) == 2
        assert results[0].node_id == sample_search_results[0].node_id

    @patch("hackathon.retrieval.reranker.Rerank")
    def test_rerank_empty_api_response(
        self, mock_rerank_class, mock_logger, mock_env_vars, sample_search_results
    ):
        """Test handling of empty API response."""
        mock_reranker = MagicMock()
        mock_reranker.generate.return_value = {"results": []}
        mock_rerank_class.return_value = mock_reranker

        reranker = WatsonxReranker()
        results = reranker.rerank("test query", sample_search_results, top_n=2)

        # Should fallback to original
        assert len(results) == 2

    @patch("hackathon.retrieval.reranker.Rerank")
    def test_rerank_document_truncation(
        self, mock_rerank_class, mock_logger, mock_env_vars
    ):
        """Test that long documents are truncated."""
        # Create result with very long text
        long_result = SearchResult(
            node_id=1,
            document_id=1,
            text_content="x" * (MAX_DOCUMENT_CHARS + 100),
            node_type="paragraph",
            node_path="chunk_0",
            score=0.85,
        )

        mock_reranker = MagicMock()
        mock_reranker.generate.return_value = {"results": [{"index": 0, "score": 0.95}]}
        mock_rerank_class.return_value = mock_reranker

        reranker = WatsonxReranker()
        reranker.rerank("test query", [long_result], top_n=1)

        # Should have called logger
        assert mock_logger.info.called

        # Should have called API with truncated document
        call_args = mock_reranker.generate.call_args
        truncated_docs = call_args[1]["inputs"]
        assert len(truncated_docs[0]) <= MAX_DOCUMENT_CHARS + 3

    @patch("hackathon.retrieval.reranker.Rerank")
    def test_rerank_short_query_warning(
        self, mock_rerank_class, mock_logger, mock_env_vars, sample_search_results
    ):
        """Test warning for short queries."""
        mock_reranker = MagicMock()
        mock_reranker.generate.return_value = {"results": [{"index": 0, "score": 0.95}]}
        mock_rerank_class.return_value = mock_reranker

        reranker = WatsonxReranker()
        reranker.rerank("pgadmin", sample_search_results, top_n=1)

        # Should have logged warning
        assert mock_logger.warning.called or mock_logger.info.called

    @patch("hackathon.retrieval.reranker.Rerank")
    def test_rerank_preserves_original_metadata(
        self, mock_rerank_class, mock_logger, mock_env_vars
    ):
        """Test that original metadata is preserved."""
        original_result = SearchResult(
            node_id=1,
            document_id=1,
            text_content="Test content",
            node_type="paragraph",
            node_path="chunk_0",
            score=0.75,
            metadata={"original_key": "original_value"},
        )

        mock_reranker = MagicMock()
        mock_reranker.generate.return_value = {"results": [{"index": 0, "score": 0.95}]}
        mock_rerank_class.return_value = mock_reranker

        reranker = WatsonxReranker()
        results = reranker.rerank("test", [original_result], top_n=1)

        # Original metadata should still be present
        assert results[0].metadata["original_key"] == "original_value"
        # New metadata should be added
        assert "original_score" in results[0].metadata
        assert "rerank_score" in results[0].metadata


class TestRerankResultsConvenience:
    """Test convenience function for reranking."""

    @patch("hackathon.retrieval.reranker.WatsonxReranker")
    def test_rerank_results_function(self, mock_reranker_class, sample_search_results):
        """Test rerank_results convenience function."""
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = sample_search_results[:1]
        mock_reranker_class.return_value = mock_reranker

        results = rerank_results("test query", sample_search_results, top_n=1)

        assert len(results) == 1
        mock_reranker_class.assert_called_once()
        mock_reranker.rerank.assert_called_once_with(
            "test query", sample_search_results, 1
        )

    @patch("hackathon.retrieval.reranker.WatsonxReranker")
    def test_rerank_results_custom_model(
        self, mock_reranker_class, sample_search_results
    ):
        """Test rerank_results with custom model."""
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = sample_search_results
        mock_reranker_class.return_value = mock_reranker

        rerank_results("test", sample_search_results, model="custom-model")

        # Should initialize with custom model
        mock_reranker_class.assert_called_once_with(model="custom-model")
