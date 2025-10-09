# Testing Guide

This guide covers writing and running tests for the RAG system.

## Test Framework 🧪

The project uses:
- **pytest** - Test runner and framework
- **pytest-cov** - Coverage reporting
- **Faker** - Generate test data
- **fixtures** - Database setup/teardown

## Running Tests ▶️

### Run All Tests

```bash
make tests
```

**Output**:
```
Running pytest...
========================= test session starts ==========================
collected 9 items

tests/test_bm25.py::test_tokenize_text PASSED                     [ 11%]
tests/test_database.py::test_create_document PASSED               [ 22%]
...

========================= 9 passed in 2.45s ============================
```

### Run Specific Test File

```bash
uv run pytest tests/test_bm25.py
```

### Run Specific Test

```bash
uv run pytest tests/test_database.py::test_create_document
```

### Run with Verbose Output

```bash
uv run pytest -v
```

### Run with Coverage Report

```bash
uv run pytest --cov=src/hackathon --cov-report=html
```

View coverage:
```bash
open htmlcov/index.html
```

---

## Test Structure 📁

```
tests/
├── conftest.py           # Shared fixtures
├── test_bm25.py          # BM25 tokenization tests
├── test_database.py      # Database operations tests
└── test_<feature>.py     # Add new test files here
```

---

## Writing Tests ✍️

### Basic Test Example

```python
# tests/test_example.py

def test_addition():
    """Test that addition works."""
    result = 2 + 2
    assert result == 4
```

Run it:
```bash
uv run pytest tests/test_example.py -v
```

### Using Fixtures

**conftest.py** provides shared fixtures:

```python
# tests/conftest.py
import pytest
from hackathon.database import SessionLocal, init_db

@pytest.fixture
def db_session():
    """Create a test database session."""
    init_db()  # Create tables
    session = SessionLocal()

    yield session

    session.close()
    # Cleanup happens here
```

**Use in tests**:

```python
# tests/test_database.py
from hackathon.database import create_document
from hackathon.models.schemas import DocumentCreate

def test_create_document(db_session):
    """Test document creation."""
    doc_data = DocumentCreate(
        filename="test.md",
        filepath="/path/to/test.md",
        metadata={"key": "value"}
    )

    doc = create_document(db_session, doc_data)

    assert doc.id is not None
    assert doc.filename == "test.md"
    assert doc.meta == {"key": "value"}
```

---

## Testing Database Operations 🗄️

### Test Document Creation

```python
# tests/test_database.py
def test_create_document(db_session):
    """Test creating a document."""
    from hackathon.database import create_document
    from hackathon.models.schemas import DocumentCreate

    doc_data = DocumentCreate(
        filename="test.md",
        filepath="/path/to/test.md",
        metadata={"title": "Test Doc"}
    )

    doc = create_document(db_session, doc_data)

    assert doc.id is not None
    assert doc.filename == "test.md"
    assert doc.filepath == "/path/to/test.md"
    assert doc.meta["title"] == "Test Doc"
```

### Test Node Hierarchy

```python
def test_node_hierarchy(db_session):
    """Test parent-child relationships."""
    from hackathon.database import create_document, create_document_node
    from hackathon.models.schemas import DocumentCreate, DocumentNodeCreate

    # Create document
    doc = create_document(db_session, DocumentCreate(
        filename="test.md",
        filepath="/test.md"
    ))

    # Create parent node
    parent = create_document_node(db_session, DocumentNodeCreate(
        document_id=doc.id,
        text_content="Parent section",
        node_type="section_header",
        node_path="section_0",
        is_leaf=False
    ))

    # Create child node
    child = create_document_node(db_session, DocumentNodeCreate(
        document_id=doc.id,
        parent_id=parent.id,
        text_content="Child paragraph",
        node_type="paragraph",
        node_path="section_0.chunk_0",
        is_leaf=True
    ))

    assert child.parent_id == parent.id
    assert child.parent.id == parent.id
    assert child.parent.text_content == "Parent section"
```

---

## Testing BM25 Tokenization 🔤

### Test Tokenizer

```python
# tests/test_bm25.py
import pytest
from hackathon.processing.bm25 import tokenize_text

@pytest.mark.parametrize("text,expected_tokens", [
    ("Hello World", ["hello", "world"]),
    ("Test-Document with-hyphens", ["test", "document", "with", "hyphens"]),
    ("CamelCase and spaces", ["camelcase", "and", "spaces"]),
    ("Numbers 123 and 456", ["numbers", "123", "and", "456"]),
])
def test_tokenize_text(text, expected_tokens):
    """Test text tokenization."""
    tokens = tokenize_text(text)
    assert tokens == expected_tokens

def test_tokenize_filters_short_tokens():
    """Test that single-character tokens are filtered."""
    text = "a b test c word d"
    tokens = tokenize_text(text)
    assert "a" not in tokens
    assert "b" not in tokens
    assert "c" not in tokens
    assert "d" not in tokens
    assert "test" in tokens
    assert "word" in tokens
```

---

## Testing Search 🔍

### Mock Embeddings

For faster tests, mock the embedding generator:

```python
# tests/test_search.py
import pytest
from unittest.mock import Mock, patch
import numpy as np

@patch('hackathon.retrieval.search.EmbeddingGenerator')
def test_semantic_search(mock_embedder_class, db_session):
    """Test semantic search with mocked embeddings."""
    from hackathon.retrieval import HybridSearcher
    from hackathon.database import create_document, create_document_node
    from hackathon.database.operations import create_embedding
    from hackathon.models.schemas import (
        DocumentCreate, DocumentNodeCreate, EmbeddingCreate
    )

    # Setup mock embedder
    mock_embedder = Mock()
    mock_embedder.embed_text.return_value = np.random.rand(384).tolist()
    mock_embedder_class.return_value = mock_embedder

    # Create test document
    doc = create_document(db_session, DocumentCreate(
        filename="test.md",
        filepath="/test.md"
    ))

    # Create test node
    node = create_document_node(db_session, DocumentNodeCreate(
        document_id=doc.id,
        text_content="Database connection pooling",
        node_type="paragraph",
        node_path="chunk_0",
        is_leaf=True
    ))

    # Create embedding
    create_embedding(db_session, EmbeddingCreate(
        node_id=node.id,
        vector=np.random.rand(384).tolist()
    ))

    # Test search
    searcher = HybridSearcher(db_session)
    results = searcher.semantic_search("pooling", top_k=5)

    assert len(results) > 0
    assert results[0][0] == node.id  # First result is our node
```

### Test Hybrid Search

```python
def test_hybrid_search(db_session):
    """Test hybrid search combining BM25 and semantic."""
    # ... setup code (create documents, nodes, embeddings, BM25 index) ...

    from hackathon.retrieval import HybridSearcher

    searcher = HybridSearcher(db_session)
    results = searcher.hybrid_search("database pooling", top_k=5)

    assert len(results) <= 5
    assert all(r.score > 0 for r in results)
    assert results[0].score >= results[1].score  # Sorted by score
```

---

## Testing Context Expansion 🌳

### Test Semantic Block Reassembly

```python
# tests/test_context_expansion.py
def test_code_block_reassembly(db_session):
    """Test reassembling split code blocks."""
    from hackathon.retrieval import ContextExpander
    from hackathon.database import create_document, create_document_node
    from hackathon.models.schemas import DocumentCreate, DocumentNodeCreate

    # Create document
    doc = create_document(db_session, DocumentCreate(
        filename="test.md",
        filepath="/test.md"
    ))

    # Create parent section
    parent = create_document_node(db_session, DocumentNodeCreate(
        document_id=doc.id,
        text_content="Configuration",
        node_type="section_header",
        node_path="section_0",
        is_leaf=False
    ))

    # Create first chunk of code
    chunk1 = create_document_node(db_session, DocumentNodeCreate(
        document_id=doc.id,
        parent_id=parent.id,
        text_content="def configure():\n    pool = Pool(",
        node_type="code",
        node_path="section_0.chunk_0",
        is_leaf=True,
        metadata={
            "headings": "Configuration",
            "docling_types": "code"
        }
    ))

    # Create second chunk of code
    chunk2 = create_document_node(db_session, DocumentNodeCreate(
        document_id=doc.id,
        parent_id=parent.id,
        text_content="        size=10\n    )",
        node_type="code",
        node_path="section_0.chunk_1",
        is_leaf=True,
        metadata={
            "headings": "Configuration",
            "docling_types": "code"
        }
    ))

    # Test reassembly
    expander = ContextExpander(db_session)
    full_code = expander.get_semantic_block_text(chunk1)

    assert full_code is not None
    assert "def configure():" in full_code
    assert "pool = Pool(" in full_code
    assert "size=10" in full_code
    assert ")" in full_code
```

### Test Hierarchical Context

```python
def test_hierarchical_context(db_session):
    """Test building hierarchical context from ancestors."""
    from hackathon.retrieval import ContextExpander
    # ... setup document tree ...

    expander = ContextExpander(db_session)
    context = expander.build_context_text(child_node, depth=2)

    assert "section_header" in context
    assert "Parent section" in context
    assert "Child paragraph" in context
```

---

## Testing Ingestion Pipeline 📝

### Test YAML Frontmatter Extraction

```python
# tests/test_ingestion.py
import tempfile
from pathlib import Path
from hackathon.processing.docling_processor import extract_yaml_frontmatter

def test_extract_yaml_frontmatter():
    """Test YAML frontmatter extraction."""
    content = """---
title: "Test Document"
date: 2024-01-15
tags: ["python", "testing"]
---

# Test Content

This is a test document.
"""

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(content)
        temp_path = Path(f.name)

    try:
        clean_content, metadata = extract_yaml_frontmatter(temp_path)

        assert "---" not in clean_content
        assert "# Test Content" in clean_content
        assert metadata["title"] == "Test Document"
        assert metadata["date"] == "2024-01-15"
    finally:
        temp_path.unlink()
```

---

## Parametrized Tests 🎯

Use `@pytest.mark.parametrize` for testing multiple inputs:

```python
import pytest

@pytest.mark.parametrize("query,expected_score_range", [
    ("database pooling", (0.7, 1.0)),  # High relevance
    ("unrelated topic", (0.0, 0.3)),   # Low relevance
    ("partial match", (0.3, 0.7)),     # Medium relevance
])
def test_search_relevance(db_session, query, expected_score_range):
    """Test search relevance for different queries."""
    # ... setup ...

    results = searcher.hybrid_search(query, top_k=1)

    if results:
        score = results[0].score
        assert expected_score_range[0] <= score <= expected_score_range[1]
```

---

## Integration Tests 🔗

Test end-to-end workflows:

```python
# tests/test_integration.py
import tempfile
from pathlib import Path

def test_end_to_end_document_processing(db_session):
    """Test complete document processing pipeline."""
    from hackathon.database import create_document
    from hackathon.processing.docling_processor import (
        extract_yaml_frontmatter,
        process_document_with_docling
    )
    from hackathon.processing.bm25 import create_bm25_index_for_node
    from hackathon.processing.embedder import EmbeddingGenerator
    from hackathon.models.schemas import DocumentCreate

    # Create test markdown file
    content = """---
title: "Integration Test"
---

# Database Pooling

Connection pooling improves performance.

```python
pool = Pool(size=10)
```
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(content)
        temp_path = Path(f.name)

    try:
        # 1. Extract frontmatter
        _, metadata = extract_yaml_frontmatter(temp_path)
        assert metadata["title"] == "Integration Test"

        # 2. Create document
        doc = create_document(db_session, DocumentCreate(
            filename=str(temp_path),
            filepath=str(temp_path),
            metadata=metadata
        ))

        # 3. Process with Docling
        node_ids = process_document_with_docling(db_session, temp_path, doc.id, metadata)
        assert len(node_ids) > 0

        # 4. Build BM25 index
        for node_id in node_ids:
            node = db_session.query(DocumentNode).get(node_id)
            create_bm25_index_for_node(db_session, node_id, node.text_content)

        # 5. Generate embeddings (mock for speed)
        # ... (in real test, use actual embedder)

        # 6. Query system
        from hackathon.retrieval import HybridSearcher
        searcher = HybridSearcher(db_session)
        results = searcher.bm25_search("pooling", top_k=5)

        assert len(results) > 0

    finally:
        temp_path.unlink()
```

---

## Mocking External Services 🎭

### Mock Anthropic LLM

```python
from unittest.mock import Mock, patch

@patch('hackathon.processing.contextual.anthropic.Anthropic')
def test_contextual_summary_generation(mock_anthropic_class, db_session):
    """Test contextual summary generation with mocked LLM."""
    # Setup mock
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock(text="This is a test summary.")]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_class.return_value = mock_client

    # ... test code ...
```

### Mock GPU/Embeddings

```python
@patch('torch.cuda.is_available')
def test_cpu_fallback(mock_cuda):
    """Test CPU fallback when GPU unavailable."""
    mock_cuda.return_value = False

    from hackathon.processing.embedder import EmbeddingGenerator

    embedder = EmbeddingGenerator()
    # Should use CPU without error
    embedding = embedder.embed_text("test")

    assert len(embedding) == 384
```

---

## Test Coverage Goals 🎯

| Component | Target Coverage |
|-----------|----------------|
| Database operations | 90%+ |
| Search (BM25 + semantic) | 85%+ |
| Context expansion | 85%+ |
| Ingestion pipeline | 80%+ |
| CLI commands | 60%+ |

Check current coverage:
```bash
uv run pytest --cov=src/hackathon --cov-report=term-missing
```

---

## Continuous Integration 🔄

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: pgvector/pgvector:pg14
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_rag_system
        ports:
          - 5432:5432

    steps:
      - uses: actions/checkout@v3

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Install dependencies
        run: uv sync --dev

      - name: Run tests
        run: uv run pytest --cov=src/hackathon
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_rag_system
```

---

## Debugging Tests 🐛

### Run with Debug Output

```bash
uv run pytest -vv -s
```

- `-vv`: Very verbose
- `-s`: Show print statements

### Use pytest Debugger

```python
def test_something():
    result = some_function()

    # Drop into debugger
    import pdb; pdb.set_trace()

    assert result == expected
```

Run:
```bash
uv run pytest --pdb
```

### Print Captured Output

```bash
# Show stdout/stderr even for passing tests
uv run pytest -rP
```

---

## Best Practices ✅

1. **One assertion per test** (when possible)
   ```python
   def test_document_creation():
       doc = create_document(...)
       assert doc.id is not None

   def test_document_filename():
       doc = create_document(...)
       assert doc.filename == "test.md"
   ```

2. **Use descriptive test names**
   ```python
   # Good ✅
   def test_hybrid_search_combines_bm25_and_semantic_scores():
       ...

   # Bad ❌
   def test_search():
       ...
   ```

3. **Use fixtures for common setup**
   ```python
   @pytest.fixture
   def sample_document(db_session):
       return create_document(db_session, DocumentCreate(...))

   def test_something(sample_document):
       # Use sample_document
       ...
   ```

4. **Clean up after tests**
   ```python
   def test_with_cleanup(db_session):
       # Create resources
       doc = create_document(...)

       try:
           # Test code
           assert ...
       finally:
           # Cleanup
           db_session.delete(doc)
           db_session.commit()
   ```

---

## Next Steps 📚

- [Contributing Guide](contributing.md) - Submit tests for review
- [Usage Guide](usage.md) - Understand what to test
- [Architecture](../architecture/overview.md) - Test critical paths
