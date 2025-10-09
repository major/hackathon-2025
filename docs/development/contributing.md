# Contributing Guide

Thank you for considering contributing to the RAG system! 🎉 This guide will help you get started.

## Code of Conduct 🤝

- Be respectful and constructive
- Welcome newcomers
- Focus on what's best for the project
- Show empathy towards others

## Getting Started 🚀

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/rag-system.git
cd rag-system
```

### 2. Set Up Development Environment

```bash
# Install dependencies
make install

# Setup pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name

# Branch naming conventions:
# - feature/add-something    (new features)
# - fix/bug-description      (bug fixes)
# - docs/update-readme       (documentation)
# - refactor/improve-search  (code improvements)
```

---

## Development Workflow 🔄

### 1. Make Changes

Edit files in your branch:
```bash
# Edit code
vim src/hackathon/retrieval/search.py

# Run tests frequently
make tests

# Check code quality
make lint
```

### 2. Write Tests

Every new feature or bug fix should include tests:

```python
# tests/test_your_feature.py
def test_new_feature(db_session):
    """Test your new feature."""
    # Arrange
    setup_data = ...

    # Act
    result = your_function(...)

    # Assert
    assert result == expected
```

Run your tests:
```bash
uv run pytest tests/test_your_feature.py -v
```

### 3. Update Documentation

If you added a feature or changed behavior:

```bash
# Update relevant documentation
vim docs/components/search.md

# Build docs locally to preview
make docs-serve
```

### 4. Commit Changes

```bash
# Stage changes
git add src/hackathon/retrieval/search.py tests/test_search.py

# Commit with clear message
git commit -m "feat: add support for custom BM25 parameters

- Allow configuring k1 and b parameters
- Add tests for parameter validation
- Update search documentation
"
```

**Commit message format**:
```
<type>: <short summary>

<optional longer description>

<optional footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

### 5. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub! 🎊

---

## Pull Request Guidelines 📋

### PR Title

Use the same format as commit messages:
```
feat: add support for custom BM25 parameters
```

### PR Description Template

```markdown
## Summary
Brief description of what this PR does.

## Changes
- Added X feature
- Fixed Y bug
- Updated Z documentation

## Testing
- [ ] All existing tests pass
- [ ] Added tests for new functionality
- [ ] Tested manually with `uv run query`

## Documentation
- [ ] Updated relevant documentation
- [ ] Added docstrings to new functions
- [ ] Updated CLAUDE.md if needed

## Screenshots (if UI changes)
[Add screenshots if applicable]

## Checklist
- [ ] Code follows project style (`make lint` passes)
- [ ] All tests pass (`make tests` passes)
- [ ] Documentation is updated
- [ ] Complexity ratings remain A-B (`make complexity`)
```

### PR Review Process

1. **Automated checks** run (tests, linting)
2. **Maintainer review** (1-2 days)
3. **Address feedback** if requested
4. **Approval and merge** 🎉

---

## Code Style Guidelines 🎨

### Python Style

We use **Ruff** for linting and formatting:

```bash
# Auto-format code
uv run ruff format src/ tests/

# Check for issues
uv run ruff check src/ tests/

# Auto-fix issues
uv run ruff check --fix src/ tests/
```

**Key conventions**:
- Use type hints
- Follow PEP 8
- Max line length: 100 characters
- Use f-strings for formatting
- Prefer comprehensions over loops

**Example**:
```python
def search_documents(
    query: str,
    top_k: int = 10,
    weights: dict[str, float] | None = None
) -> list[SearchResult]:
    """
    Search for documents matching the query.

    Args:
        query: Search query string
        top_k: Number of results to return
        weights: Optional score weights (bm25, semantic)

    Returns:
        List of SearchResult objects sorted by relevance

    Raises:
        ValueError: If top_k is less than 1
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    weights = weights or {"bm25": 0.4, "semantic": 0.6}

    # Use comprehension instead of loop
    results = [
        SearchResult(node_id=node.id, score=score)
        for node, score in self._compute_scores(query)
    ]

    return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]
```

### Complexity Guidelines

Keep functions simple and readable:

```bash
# Check complexity
make complexity
```

**Targets**:
- **A rating**: Ideal (complexity 1-5)
- **B rating**: Acceptable (complexity 6-10)
- **C rating**: Consider refactoring (complexity 11-20)
- **D/F rating**: Must refactor before merging (complexity 21+)

**Refactoring example**:

```python
# ❌ Bad: D rating (complexity 27)
def process_document(doc):
    # 50+ lines of nested ifs and loops
    if condition1:
        for item in items:
            if condition2:
                while condition3:
                    # ... more nesting
                    pass

# ✅ Good: A/B rating
def process_document(doc):
    """Main processing function."""
    metadata = _extract_metadata(doc)
    chunks = _create_chunks(doc)
    return _index_chunks(chunks, metadata)

def _extract_metadata(doc):
    """Extract document metadata."""
    # Simple focused function
    ...

def _create_chunks(doc):
    """Create document chunks."""
    # Simple focused function
    ...

def _index_chunks(chunks, metadata):
    """Index chunks in database."""
    # Simple focused function
    ...
```

---

## Testing Requirements ✅

### Test Coverage

All new code must include tests:

```bash
# Check coverage
uv run pytest --cov=src/hackathon --cov-report=term-missing

# Coverage targets:
# - New features: 85%+ coverage
# - Bug fixes: Test reproducing the bug + fix verification
```

### Test Quality

**Good test**:
```python
def test_hybrid_search_combines_scores_correctly():
    """Test that hybrid search properly weights BM25 and semantic scores."""
    # Arrange
    db = setup_test_database()
    searcher = HybridSearcher(db)

    # Act
    results = searcher.hybrid_search(
        "test query",
        bm25_weight=0.4,
        semantic_weight=0.6
    )

    # Assert
    assert len(results) > 0
    # Check that score is weighted combination
    first_result = results[0]
    expected_score = (0.4 * bm25_score) + (0.6 * semantic_score)
    assert abs(first_result.score - expected_score) < 0.01
```

**Bad test**:
```python
def test_search():
    """Test search."""  # Vague description
    results = search("test")  # No setup
    assert results  # Weak assertion
```

---

## Documentation Standards 📚

### Docstrings

Use Google-style docstrings:

```python
def hybrid_search(
    self,
    query: str,
    top_k: int = 10,
    bm25_weight: float = 0.4,
    semantic_weight: float = 0.6,
) -> list[SearchResult]:
    """
    Perform hybrid search combining BM25 and semantic similarity.

    This function merges results from keyword-based BM25 search and
    vector-based semantic search, combining their scores using
    configurable weights.

    Args:
        query: User's search query
        top_k: Number of results to return (default: 10)
        bm25_weight: Weight for BM25 scores, 0-1 (default: 0.4)
        semantic_weight: Weight for semantic scores, 0-1 (default: 0.6)

    Returns:
        List of SearchResult objects, sorted by combined score descending

    Raises:
        ValueError: If weights don't sum to 1.0 or top_k < 1

    Example:
        >>> searcher = HybridSearcher(db)
        >>> results = searcher.hybrid_search("database pooling", top_k=5)
        >>> for result in results:
        ...     print(f"{result.score:.4f}: {result.text_content[:50]}")
        0.9234: Connection pooling reduces overhead by...
    """
    ...
```

### Inline Comments

Use sparingly, only for complex logic:

```python
# ✅ Good: Explains WHY
# Normalize BM25 scores to 0-1 range to match semantic scores
normalized_bm25 = bm25_score / max_bm25_score

# ❌ Bad: Explains WHAT (obvious from code)
# Add 1 to the counter
counter += 1
```

---

## Common Contribution Areas 🎯

### 1. Add New Embedding Model Support

**File**: `src/hackathon/processing/embedder.py`

```python
# Add model configuration
SUPPORTED_MODELS = {
    "granite": "ibm-granite/granite-embedding-30m-english",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "your-model": "author/model-name",  # Add here
}
```

**Don't forget**:
- Update docs
- Add tests
- Update `.env.example`

### 2. Improve Search Ranking

**File**: `src/hackathon/retrieval/search.py`

Ideas:
- Add query expansion
- Implement re-ranking
- Add result diversity

**Remember**:
- Benchmark before/after
- Write tests
- Document changes

### 3. Add CLI Commands

**File**: `src/hackathon/cli/new_command.py`

```python
"""Description of new command."""

from rich.console import Console

console = Console()

def main() -> None:
    """Execute the new command."""
    console.print("[bold blue]Running new command...[/bold blue]")
    # Your implementation
    ...

if __name__ == "__main__":
    main()
```

**Update**: `pyproject.toml`

```toml
[project.scripts]
new-command = "hackathon.cli.new_command:main"
```

### 4. Fix Bugs

Found a bug? Great! 🐛

1. **Create an issue** describing the bug
2. **Write a failing test** that reproduces it
3. **Fix the bug**
4. **Verify test passes**
5. **Submit PR** with issue reference

---

## Advanced Topics 🔬

### Adding a New Database Table

1. **Define model** (`src/hackathon/models/database.py`):
```python
class NewTable(Base):
    __tablename__ = "new_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
```

2. **Add schema** (`src/hackathon/models/schemas.py`):
```python
class NewTableCreate(BaseModel):
    name: str

class NewTableRead(BaseModel):
    id: int
    name: str
    created_at: datetime
```

3. **Add operations** (`src/hackathon/database/operations.py`):
```python
def create_new_table_entry(db: Session, data: NewTableCreate) -> NewTable:
    entry = NewTable(**data.model_dump())
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry
```

4. **Add migration** (manual for now):
```bash
# Users must run: uv run clean
# Future: Add alembic migrations
```

### Optimizing Vector Search

**File**: `src/hackathon/database/connection.py`

```python
# Tune HNSW index parameters
def init_db():
    # ... existing code ...

    # Add optimized vector index
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS embeddings_vector_idx
            ON embeddings USING hnsw (vector vector_cosine_ops)
            WITH (m = 32, ef_construction = 128);
        """))
```

**Before committing**:
- Benchmark query performance
- Test with large datasets (10K+ documents)
- Document trade-offs in PR

---

## Getting Help 🆘

### Where to Ask Questions

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Pull Request Comments**: Code-specific questions

### Useful Resources

- [Project Architecture](../architecture/overview.md)
- [Development Setup](setup.md)
- [Testing Guide](testing.md)
- [Usage Guide](usage.md)

---

## Recognition 🏆

Contributors are recognized in:

1. **README.md** - Contributors section
2. **Changelog** - Release notes
3. **Git history** - Your commits!

Thank you for contributing! 🙏

---

## Checklist for First-Time Contributors ✅

- [ ] Forked and cloned repository
- [ ] Set up development environment (`make install`)
- [ ] Created feature branch
- [ ] Made changes with tests
- [ ] All checks pass (`make all`)
- [ ] Updated documentation
- [ ] Committed with clear messages
- [ ] Pushed to fork
- [ ] Created Pull Request
- [ ] Responded to review feedback

**Welcome to the project!** 🎉

---

## Quick Command Reference 📋

```bash
# Setup
make install              # Install dependencies
git checkout -b feat/X    # Create branch

# Development
make tests                # Run tests
make lint                 # Check style
make complexity           # Check complexity
make all                  # Run all checks

# Commit
git add .
git commit -m "feat: description"
git push origin feat/X

# Documentation
make docs-serve           # Preview docs
make docs-build           # Build docs

# Database
uv run clean              # Reset database
uv run process            # Rebuild index
```

Happy coding! 🚀
