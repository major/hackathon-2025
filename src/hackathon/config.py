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
    db_host: str = Field(default="127.0.0.1", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_user: str = Field(default="postgres", alias="DB_USER")
    db_password: str = Field(default="secrete", alias="DB_PASSWORD")
    db_name: str = Field(default="hackathon", alias="DB_NAME")

    # LLM API Configuration
    llm_api_base: str = Field(default="http://127.0.0.1:8086/v1", alias="LLM_API_BASE")
    llm_model: str = Field(default="granite4", alias="LLM_MODEL")
    llm_api_key: str = Field(default="not-needed", alias="LLM_API_KEY")

    # Embedding Configuration
    embedding_model: str = Field(
        default="ibm-granite/granite-embedding-30m-english", alias="EMBEDDING_MODEL"
    )
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")

    # Processing Configuration
    markdown_directory: str = Field(
        default="blog/content/posts", alias="MARKDOWN_DIRECTORY"
    )
    markdown_pattern: str = Field(default="**/*.md", alias="MARKDOWN_PATTERN")

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get application settings instance (cached singleton).

    Returns:
        Configured Settings instance (cached for performance)
    """
    return Settings()
