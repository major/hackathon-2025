"""Pytest configuration and fixtures."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from hackathon.models.database import Base


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
    session.close()


@pytest.fixture
def sample_markdown():
    """Sample markdown content for testing."""
    return """---
title: Test Document
date: 2025-10-08
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
