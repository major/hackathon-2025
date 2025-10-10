"""
Vulture whitelist for known false positives.

This file tells vulture about code that appears unused but is actually used:
- CLI entry points defined in pyproject.toml
- Pydantic model fields (accessed via __dict__ or model_dump)
- SQLAlchemy ORM columns and relationships
- Test fixtures
- __init__.py exports
"""

# CLI entry points (defined in pyproject.toml [project.scripts])
# These functions are called by uv/pip when running `uv run process`, etc.
_.main  # All main() functions in cli/ modules

# Pydantic model fields that look unused but are accessed via validation
_.model_config  # Pydantic model configuration
_.model_dump  # Pydantic serialization
_.model_validate  # Pydantic deserialization

# SQLAlchemy ORM - columns and relationships accessed via attribute access
_.meta  # Mapped to 'metadata' column in database
_.nodes  # SQLAlchemy relationship
_.document  # SQLAlchemy relationship
_.parent  # SQLAlchemy relationship
_.children  # SQLAlchemy relationship
_.embedding  # SQLAlchemy relationship
_.contextual_chunk  # SQLAlchemy relationship
_.node  # SQLAlchemy relationship

# __init__.py exports (re-exported for convenience)
_.get_db  # Re-exported in __init__.py
_.ContextExpander  # Re-exported in __init__.py
_.HybridSearcher  # Re-exported in __init__.py

# Test fixtures (used by pytest via name matching)
_.db  # pytest fixture
_.sample_document  # pytest fixture
