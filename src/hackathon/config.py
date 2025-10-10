"""Configuration management using Pydantic Settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Database Configuration
    db_host: str = Field(
        default="127.0.0.1",
        alias="DB_HOST",
    )
    db_port: int = Field(
        default=5432,
        alias="DB_PORT",
    )
    db_user: str = Field(
        default="postgres",
        alias="DB_USER",
    )
    db_password: str = Field(
        default="secrete",
        alias="DB_PASSWORD",
    )
    db_name: str = Field(
        default="hackathon",
        alias="DB_NAME",
    )

    # Processing Configuration
    markdown_directory: str = Field(
        default="blog/content/posts",
        alias="MARKDOWN_DIRECTORY",
    )
    markdown_pattern: str = Field(
        default="**/*.md",
        alias="MARKDOWN_PATTERN",
    )

    # BM25S Index Configuration
    bm25_index_path: str = Field(
        default=".bm25s_index",
        alias="BM25_INDEX_PATH",
    )

    # IBM Watsonx Configuration (for contextual retrieval and reranking)
    watsonx_api_key: str = Field(
        default="",
        alias="WATSONX_API_KEY",
    )
    watsonx_project_id: str = Field(
        default="",
        alias="WATSONX_PROJECT_ID",
    )
    watsonx_url: str = Field(
        default="https://us-south.ml.cloud.ibm.com",
        alias="WATSONX_URL",
    )
    # LLM model for contextual summaries
    watsonx_llm_model: str = Field(
        default="ibm/granite-4-h-small",
        alias="WATSONX_LLM_MODEL",
    )
    # Reranker model
    watsonx_reranker_model: str = Field(
        default="cross-encoder/ms-marco-minilm-l-12-v2",
        alias="WATSONX_RERANKER_MODEL",
    )

    @property
    def database_url(self) -> str:
        """
        Construct PostgreSQL database URL.

        Returns:
            Database connection URL string
        """
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


class ConfigValidator:
    """Validate configuration settings at startup."""

    @staticmethod
    def validate_watsonx_credentials(settings: Settings) -> bool:
        """
        Check if Watsonx credentials are configured.

        Args:
            settings: Application settings instance

        Returns:
            True if both API key and project ID are set
        """
        return bool(settings.watsonx_api_key and settings.watsonx_project_id)

    @staticmethod
    def validate_database_connection(settings: Settings) -> tuple[bool, str]:
        """
        Validate database connection settings (without actually connecting).

        Args:
            settings: Application settings instance

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not settings.db_host:
            return False, "DB_HOST is not configured"
        if not settings.db_name:
            return False, "DB_NAME is not configured"
        if not settings.db_user:
            return False, "DB_USER is not configured"
        if not settings.db_password:
            return False, "DB_PASSWORD is not configured"
        if settings.db_port < 1 or settings.db_port > 65535:
            return False, f"DB_PORT {settings.db_port} is out of valid range (1-65535)"
        return True, ""

    @staticmethod
    def validate_paths(settings: Settings) -> tuple[bool, str]:
        """
        Validate file system paths exist and are accessible.

        Args:
            settings: Application settings instance

        Returns:
            Tuple of (is_valid, error_message)
        """
        from pathlib import Path

        # Check markdown directory exists
        md_dir = Path(settings.markdown_directory)
        if not md_dir.exists():
            return (
                False,
                f"Markdown directory does not exist: {settings.markdown_directory}",
            )
        if not md_dir.is_dir():
            return (
                False,
                f"Markdown path is not a directory: {settings.markdown_directory}",
            )

        return True, ""

    @classmethod
    def validate_all(cls, settings: Settings, strict: bool = False) -> list[str]:
        """
        Validate all configuration settings.

        Args:
            settings: Application settings instance
            strict: If True, treat warnings as errors (e.g., missing Watsonx creds)

        Returns:
            List of error/warning messages (empty if all valid)
        """
        errors = []

        # Database validation (always required)
        db_valid, db_error = cls.validate_database_connection(settings)
        if not db_valid:
            errors.append(f"Database configuration error: {db_error}")

        # Path validation (only if directory is not default)
        if settings.markdown_directory != "blog/content/posts":
            path_valid, path_error = cls.validate_paths(settings)
            if not path_valid:
                errors.append(f"Path configuration error: {path_error}")

        # Watsonx validation (optional unless strict mode)
        if not cls.validate_watsonx_credentials(settings):
            msg = "Watsonx credentials not configured (contextual retrieval and reranking will be unavailable)"
            if strict:
                errors.append(f"Watsonx configuration error: {msg}")
            # In non-strict mode, this is just a warning (not added to errors)

        return errors


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get application settings instance (cached singleton).

    Returns:
        Configured Settings instance (cached for performance)
    """
    return Settings()
