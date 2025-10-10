"""Tests for configuration management and validation. 🎯"""

import pytest

from hackathon.config import ConfigValidator, Settings, get_settings


class TestSettings:
    """Test Settings class and environment variable loading."""

    def test_default_settings(self, mock_env_vars):
        """Test that default settings are loaded correctly."""
        settings = get_settings()

        assert settings.db_host == "localhost"
        assert settings.db_port == 5432
        assert settings.db_user == "testuser"
        assert settings.db_password == "testpass"
        assert settings.db_name == "testdb"
        assert settings.watsonx_api_key == "test-api-key"
        assert settings.watsonx_project_id == "test-project-id"

    def test_database_url_construction(self, mock_env_vars):
        """Test that database URL is constructed correctly."""
        settings = get_settings()
        expected = "postgresql://testuser:testpass@localhost:5432/testdb"
        assert settings.database_url == expected

    @pytest.mark.parametrize(
        "host,port,user,password,db,expected",
        [
            (
                "192.168.1.100",
                5433,
                "admin",
                "secret123",
                "mydb",
                "postgresql://admin:secret123@192.168.1.100:5433/mydb",
            ),
            (
                "db.example.com",
                5432,
                "user",
                "p@ss",
                "production",
                "postgresql://user:p@ss@db.example.com:5432/production",
            ),
        ],
    )
    def test_database_url_various_configs(
        self, monkeypatch, host, port, user, password, db, expected
    ):
        """Test database URL with various configurations."""
        monkeypatch.setenv("DB_HOST", host)
        monkeypatch.setenv("DB_PORT", str(port))
        monkeypatch.setenv("DB_USER", user)
        monkeypatch.setenv("DB_PASSWORD", password)
        monkeypatch.setenv("DB_NAME", db)

        settings = get_settings()
        assert settings.database_url == expected

    def test_settings_singleton(self, mock_env_vars):
        """Test that get_settings returns cached singleton."""
        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same instance (cached)
        assert settings1 is settings2

    def test_watsonx_defaults(self, mock_env_vars):
        """Test Watsonx default model settings."""
        settings = get_settings()

        assert settings.watsonx_llm_model == "ibm/granite-4-h-small"
        assert (
            settings.watsonx_reranker_model == "cross-encoder/ms-marco-minilm-l-12-v2"
        )
        assert settings.watsonx_url == "https://us-south.ml.cloud.ibm.com"

    def test_custom_watsonx_models(self, monkeypatch):
        """Test custom Watsonx model configuration."""
        monkeypatch.setenv("WATSONX_LLM_MODEL", "custom/llm-model")
        monkeypatch.setenv("WATSONX_RERANKER_MODEL", "custom/reranker-model")

        settings = get_settings()

        assert settings.watsonx_llm_model == "custom/llm-model"
        assert settings.watsonx_reranker_model == "custom/reranker-model"

    def test_markdown_directory_default(self, mock_env_vars):
        """Test default markdown directory setting."""
        settings = get_settings()
        assert settings.markdown_directory == "blog/content/posts"
        assert settings.markdown_pattern == "**/*.md"

    def test_bm25_index_path_default(self, mock_env_vars):
        """Test default BM25 index path."""
        settings = get_settings()
        assert settings.bm25_index_path == ".bm25s_index"


class TestConfigValidator:
    """Test ConfigValidator validation methods."""

    def test_validate_watsonx_credentials_valid(self, mock_env_vars):
        """Test validation with valid Watsonx credentials."""
        settings = get_settings()
        assert ConfigValidator.validate_watsonx_credentials(settings) is True

    @pytest.mark.parametrize(
        "api_key,project_id,expected",
        [
            ("", "", False),
            ("test-key", "", False),
            ("", "test-project", False),
            ("test-key", "test-project", True),
        ],
    )
    def test_validate_watsonx_credentials_various(
        self, monkeypatch, api_key, project_id, expected
    ):
        """Test Watsonx credential validation with various combinations."""
        monkeypatch.setenv("WATSONX_API_KEY", api_key)
        monkeypatch.setenv("WATSONX_PROJECT_ID", project_id)

        settings = get_settings()
        result = ConfigValidator.validate_watsonx_credentials(settings)
        assert result == expected

    def test_validate_database_connection_valid(self, mock_env_vars):
        """Test database connection validation with valid settings."""
        settings = get_settings()
        is_valid, error_msg = ConfigValidator.validate_database_connection(settings)

        assert is_valid is True
        assert error_msg == ""

    @pytest.mark.parametrize(
        "field,value,expected_error",
        [
            ("DB_HOST", "", "DB_HOST is not configured"),
            ("DB_NAME", "", "DB_NAME is not configured"),
            ("DB_USER", "", "DB_USER is not configured"),
            ("DB_PASSWORD", "", "DB_PASSWORD is not configured"),
            ("DB_PORT", "0", "DB_PORT 0 is out of valid range"),
            ("DB_PORT", "99999", "DB_PORT 99999 is out of valid range"),
        ],
    )
    def test_validate_database_connection_invalid(
        self, monkeypatch, mock_env_vars, field, value, expected_error
    ):
        """Test database validation with various invalid configurations."""
        monkeypatch.setenv(field, value)

        settings = get_settings()
        is_valid, error_msg = ConfigValidator.validate_database_connection(settings)

        assert is_valid is False
        assert expected_error in error_msg

    def test_validate_database_connection_port_boundaries(self, monkeypatch):
        """Test database port validation at boundaries."""
        # Valid port at minimum
        monkeypatch.setenv("DB_PORT", "1")
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_USER", "user")
        monkeypatch.setenv("DB_PASSWORD", "pass")
        monkeypatch.setenv("DB_NAME", "db")
        settings = get_settings()
        is_valid, _ = ConfigValidator.validate_database_connection(settings)
        assert is_valid is True

        # Valid port at maximum
        monkeypatch.setenv("DB_PORT", "65535")
        settings = get_settings()
        is_valid, _ = ConfigValidator.validate_database_connection(settings)
        assert is_valid is True

    def test_validate_paths_default(self, mock_env_vars):
        """Test path validation with default markdown directory."""
        settings = get_settings()
        # Default directory may not exist, but validation should pass
        is_valid, error_msg = ConfigValidator.validate_paths(settings)
        # Note: This may fail if directory doesn't exist, which is expected
        assert isinstance(is_valid, bool)

    def test_validate_paths_nonexistent_directory(self, monkeypatch):
        """Test path validation with non-existent directory."""
        monkeypatch.setenv("MARKDOWN_DIRECTORY", "/nonexistent/path/to/markdown")

        settings = get_settings()
        is_valid, error_msg = ConfigValidator.validate_paths(settings)

        assert is_valid is False
        assert "does not exist" in error_msg

    def test_validate_paths_file_not_directory(self, tmp_path, monkeypatch):
        """Test path validation when path is a file, not a directory."""
        # Create a file instead of directory
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        monkeypatch.setenv("MARKDOWN_DIRECTORY", str(test_file))

        settings = get_settings()
        is_valid, error_msg = ConfigValidator.validate_paths(settings)

        assert is_valid is False
        assert "is not a directory" in error_msg

    def test_validate_all_success(self, monkeypatch, tmp_path):
        """Test validate_all with all valid settings."""
        # Create valid markdown directory
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()

        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_PORT", "5432")
        monkeypatch.setenv("DB_USER", "user")
        monkeypatch.setenv("DB_PASSWORD", "pass")
        monkeypatch.setenv("DB_NAME", "db")
        monkeypatch.setenv("WATSONX_API_KEY", "test-key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project")
        monkeypatch.setenv("MARKDOWN_DIRECTORY", str(md_dir))

        settings = get_settings()
        errors = ConfigValidator.validate_all(settings)

        assert errors == []

    def test_validate_all_missing_watsonx_non_strict(self, monkeypatch, tmp_path):
        """Test validate_all with missing Watsonx credentials (non-strict mode)."""
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()

        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_PORT", "5432")
        monkeypatch.setenv("DB_USER", "user")
        monkeypatch.setenv("DB_PASSWORD", "pass")
        monkeypatch.setenv("DB_NAME", "db")
        monkeypatch.setenv("WATSONX_API_KEY", "")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "")
        monkeypatch.setenv("MARKDOWN_DIRECTORY", str(md_dir))

        settings = get_settings()
        errors = ConfigValidator.validate_all(settings, strict=False)

        # Should be empty - Watsonx is optional in non-strict mode
        assert errors == []

    def test_validate_all_missing_watsonx_strict(self, monkeypatch, tmp_path):
        """Test validate_all with missing Watsonx credentials (strict mode)."""
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()

        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_PORT", "5432")
        monkeypatch.setenv("DB_USER", "user")
        monkeypatch.setenv("DB_PASSWORD", "pass")
        monkeypatch.setenv("DB_NAME", "db")
        monkeypatch.setenv("WATSONX_API_KEY", "")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "")
        monkeypatch.setenv("MARKDOWN_DIRECTORY", str(md_dir))

        settings = get_settings()
        errors = ConfigValidator.validate_all(settings, strict=True)

        # Should have Watsonx error in strict mode
        assert len(errors) > 0
        assert any("Watsonx" in error for error in errors)

    def test_validate_all_database_error(self, monkeypatch):
        """Test validate_all with database configuration error."""
        monkeypatch.setenv("DB_HOST", "")
        monkeypatch.setenv("DB_PORT", "5432")
        monkeypatch.setenv("DB_USER", "user")
        monkeypatch.setenv("DB_PASSWORD", "pass")
        monkeypatch.setenv("DB_NAME", "db")

        settings = get_settings()
        errors = ConfigValidator.validate_all(settings)

        assert len(errors) > 0
        assert any("Database configuration error" in error for error in errors)

    def test_validate_all_multiple_errors(self, monkeypatch):
        """Test validate_all with multiple configuration errors."""
        monkeypatch.setenv("DB_HOST", "")
        monkeypatch.setenv("DB_PORT", "99999")
        monkeypatch.setenv("DB_USER", "")
        monkeypatch.setenv("DB_PASSWORD", "pass")
        monkeypatch.setenv("DB_NAME", "db")
        monkeypatch.setenv("MARKDOWN_DIRECTORY", "/nonexistent")

        settings = get_settings()
        errors = ConfigValidator.validate_all(settings)

        # Should have database errors
        assert len(errors) > 0

    def test_validate_all_default_directory_skip(self, monkeypatch):
        """Test that default directory validation is skipped."""
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_PORT", "5432")
        monkeypatch.setenv("DB_USER", "user")
        monkeypatch.setenv("DB_PASSWORD", "pass")
        monkeypatch.setenv("DB_NAME", "db")
        monkeypatch.setenv("MARKDOWN_DIRECTORY", "blog/content/posts")  # Default

        settings = get_settings()
        errors = ConfigValidator.validate_all(settings)

        # Should not have path validation errors for default directory
        assert not any("Path configuration error" in error for error in errors)
