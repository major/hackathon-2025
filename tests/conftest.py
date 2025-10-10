"""Pytest configuration and fixtures."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from hackathon.models.database import Base, Document, DocumentNode

# Disable Rich formatting in tests to avoid ANSI color codes
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"


@pytest.fixture
def db_engine():
    """Create a test database engine."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def db_session(db_engine) -> Session:
    """Create a test database session."""
    session_local = sessionmaker(bind=db_engine)
    session = session_local()
    yield session
    session.rollback()  # Rollback any uncommitted changes
    session.close()


@pytest.fixture
def sample_markdown():
    """Sample markdown content for testing."""
    return """---
title: Test Document
date: 2025-10-08
tags: [test, python]
---

# Introduction

This is a test document with multiple sections.

## Section 1

This is the first section with some content.

### Subsection 1.1

More detailed content in a subsection.

## Section 2

This is the second section.

It has multiple paragraphs.
"""


@pytest.fixture
def sample_markdown_no_frontmatter():
    """Sample markdown without frontmatter."""
    return """# Simple Document

This is a simple document without frontmatter.

## Section

Some content here.
"""


@pytest.fixture
def sample_markdown_with_code():
    """Sample markdown with code blocks for testing."""
    return """---
title: Code Example
---

# Setup Guide

Install dependencies:

```bash
npm install
pip install -r requirements.txt
```

Configure the database in `.env`:

```
DB_HOST=localhost
DB_PORT=5432
```
"""


@pytest.fixture
def temp_markdown_file(sample_markdown):
    """Create a temporary markdown file for testing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as tmp_file:
        tmp_file.write(sample_markdown)
        tmp_path = Path(tmp_file.name)

    yield tmp_path

    # Cleanup
    tmp_path.unlink(missing_ok=True)


@pytest.fixture
def sample_document(db_session: Session) -> Document:
    """Create a sample document in the database."""
    doc = Document(
        filename="test.md",
        filepath="/path/to/test.md",
        meta={"title": "Test Document", "date": "2025-10-08"},
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    return doc


@pytest.fixture
def sample_nodes(db_session: Session, sample_document: Document) -> list[DocumentNode]:
    """Create sample document nodes with sequential positions."""
    nodes = []
    texts = [
        "First paragraph about introduction.",
        "Second paragraph with more details.",
        "Third paragraph with examples.",
        "Fourth paragraph as conclusion.",
    ]

    for i, text in enumerate(texts):
        node = DocumentNode(
            document_id=sample_document.id,
            node_type="paragraph",
            text_content=text,
            is_leaf=True,
            node_path=f"chunk_{i}",
            position=i,
            meta={"headings": f"Section {i // 2 + 1}"},
        )
        db_session.add(node)
        nodes.append(node)

    db_session.commit()
    for node in nodes:
        db_session.refresh(node)

    return nodes


@pytest.fixture
def mock_watsonx_credentials():
    """Mock Watsonx credentials for testing."""
    return {
        "api_key": "test-api-key-12345",
        "project_id": "test-project-id-67890",
        "url": "https://us-south.ml.cloud.ibm.com",
    }


@pytest.fixture
def mock_watsonx_model():
    """Mock Watsonx ModelInference for testing."""
    mock = MagicMock()
    mock.generate_text.return_value = (
        "This section explains the setup process for the application."
    )
    return mock


@pytest.fixture
def mock_watsonx_reranker():
    """Mock Watsonx Rerank API response."""
    mock = MagicMock()
    mock.generate.return_value = {
        "results": [
            {"index": 2, "score": 0.95},
            {"index": 0, "score": 0.87},
            {"index": 1, "score": 0.72},
        ]
    }
    return mock


@pytest.fixture
def sample_search_results():
    """Sample SearchResult objects for testing reranking."""
    from hackathon.models.schemas import SearchResult

    return [
        SearchResult(
            node_id=1,
            document_id=1,
            text_content="First result about Python programming.",
            node_type="paragraph",
            node_path="chunk_0",
            score=0.85,
            metadata={"headings": "Programming > Python"},
        ),
        SearchResult(
            node_id=2,
            document_id=1,
            text_content="Second result about database configuration.",
            node_type="paragraph",
            node_path="chunk_1",
            score=0.72,
            metadata={"headings": "Setup > Database"},
        ),
        SearchResult(
            node_id=3,
            document_id=2,
            text_content="Third result about testing strategies.",
            node_type="paragraph",
            node_path="chunk_2",
            score=0.68,
            metadata={"headings": "Development > Testing"},
        ),
    ]


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set mock environment variables for testing."""
    env_vars = {
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_USER": "testuser",
        "DB_PASSWORD": "testpass",
        "DB_NAME": "testdb",
        "WATSONX_API_KEY": "test-api-key",
        "WATSONX_PROJECT_ID": "test-project-id",
        "WATSONX_URL": "https://us-south.ml.cloud.ibm.com",
        "MARKDOWN_DIRECTORY": "blog/content/posts",
        "BM25_INDEX_PATH": ".bm25s_index",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Reset the settings cache between tests."""
    from hackathon.config import get_settings

    # Clear the LRU cache to avoid settings pollution between tests
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
