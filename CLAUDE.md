# Claude Code Guide - RAG System

This document provides context for Claude Code instances working in this repository.

## Quick Commands

```bash
# Document processing pipeline
uv run process         # Process all markdown files (parse, chunk, index, embed, contextualize)
uv run query          # Interactive query interface
uv run query "text"   # Direct query
uv run clean          # Drop all database tables and start fresh

# Development
make tests            # Run pytest suite
make lint             # Run ruff linter and formatter
make types            # Run pyright type checker
make complexity       # Run radon code complexity analysis
make all              # Run all checks (lint, types, tests)
make install          # Install dependencies with uv

# Documentation
make docs-serve       # Start MkDocs dev server with live reload (http://127.0.0.1:8000)
make docs-build       # Build MkDocs static site for production (outputs to site/)
```

## High-Level Architecture

This is a proof-of-concept RAG system that combines multiple retrieval strategies for improved accuracy.

### Document Processing Pipeline

```
Markdown files
    ↓
1. Extract YAML frontmatter (python-frontmatter library)
    ↓
2. Docling DocumentConverter (parse document structure from clean content)
    ↓
3. Docling HybridChunker (intelligent chunking with hierarchy)
    ↓
4. Create DocumentNodes in PostgreSQL
    ├── BM25 tokenization → bm25_index table
    ├── Embedding generation (GPU) → embeddings table (pgvector)
    └── Contextual summaries (LLM) → contextual_chunks table
```

### Query Pipeline

```
User query
    ↓
Hybrid Search (parallel):
    ├── BM25 keyword search (40% weight)
    └── Semantic similarity search (60% weight)
    ↓
Combined ranking
    ↓
Context expansion (optional):
    ├── Semantic block reassembly (lists, code blocks, tables)
    └── Hierarchical tree traversal (parent/sibling nodes)
    ↓
Results with expanded context
```

### Core Components

**Processing (`src/hackathon/processing/`)**:
- `docling_processor.py`: Docling integration for document parsing and chunking
  - `extract_yaml_frontmatter()`: YAML parsing using python-frontmatter library
  - `process_document_with_docling()`: Creates temp file without frontmatter for Docling processing
- `bm25.py`: BM25 tokenization and indexing (rank-bm25 library)
- `embedder.py`: GPU-accelerated embeddings (sentence-transformers + granite-embedding-30m-english)
- `contextual.py`: LLM-based contextual summaries (Anthropic's Contextual Retrieval pattern)

**Retrieval (`src/hackathon/retrieval/`)**:
- `search.py`: `HybridSearcher` class combining BM25 + semantic search
- `context_expansion.py`: `ContextExpander` class for tree traversal and semantic block reassembly

**Database (`src/hackathon/database/`)**:
- `connection.py`: SQLAlchemy setup with pgvector extension + `session_scope()` context manager
- `operations.py`: CRUD operations for documents, nodes, embeddings, etc.

**Models (`src/hackathon/models/`)**:
- `database.py`: SQLAlchemy ORM models (Document, DocumentNode, Embedding, BM25Index, ContextualChunk)
- `schemas.py`: Pydantic schemas for validation and API contracts

## Key Technical Details

### Database Schema

- **documents**: Source file metadata
- **document_nodes**: Hierarchical tree structure with `parent_id` relationships
  - `node_type`: From Docling (e.g., "paragraph", "list_item", "code", "section_header", "table")
  - `is_leaf`: Boolean flag for indexable chunks
  - `metadata`: JSON field with Docling metadata (headings, docling_types, origin)
- **embeddings**: pgvector HNSW index for semantic similarity (384-dim vectors)
- **bm25_index**: Tokenized terms for keyword search
- **contextual_chunks**: LLM-generated contextual summaries

### Docling Integration

Docling provides:
- Automatic document structure detection (headings, lists, code blocks, tables)
- HybridChunker for intelligent chunking that respects semantic boundaries
- Rich metadata per chunk:
  - `doc_items`: List of content types in this chunk
  - `headings`: Hierarchical heading context (e.g., "System Logging, Using journalctl")
  - `origin`: Document elements this chunk came from

Important: YAML frontmatter is extracted using the python-frontmatter library. The clean content (without frontmatter) is written to a temporary file that Docling processes, ensuring frontmatter doesn't interfere with document structure detection.

### BM25 Indexing

BM25 scoring uses TF-IDF with document length normalization. The in-memory index is rebuilt on every query:
1. Load all BM25Index rows for corpus
2. Build rank-bm25 BM25Okapi object
3. Tokenize query
4. Score all documents
5. Return top-k results

This approach is simple but doesn't scale beyond ~10k documents. Consider switching to Tantivy or similar for production.

### Hybrid Search Weights

Current weights (in `search.py`):
- BM25: 40%
- Semantic: 60%

These can be adjusted based on your use case. BM25 is better for exact keyword matches; semantic is better for conceptual similarity.

### Contextual Retrieval

Implements Anthropic's Contextual Retrieval pattern:
1. For each chunk, generate a contextual summary using an LLM
2. Store summary in separate table (`contextual_chunks`)
3. During search, optionally include contextual summary for better relevance

Current implementation uses JSON mode with post-processing using Python's `str.removeprefix()` to strip common prefixes ("Context:", "Summary:") that LLMs sometimes add.

### Semantic Block Reassembly

When a chunk is part of a larger semantic block (list, code block, table), the system can reassemble the full block by:
1. Checking Docling's `docling_types` metadata
2. Finding all sibling chunks with same parent
3. Reconstructing the complete block

This prevents returning "item 5 in a 10-step process" without context.

## Important Patterns

### Pydantic Settings

Configuration is managed via `src/hackathon/config.py` using pydantic-settings:
- Loads from `.env` file
- Type validation
- Singleton pattern via `@lru_cache`

### Centralized Logging

Logging is configured in `src/hackathon/utils/logging.py` using rich:
- Use `get_logger(__name__)` to get a logger instance
- Rich formatting with tracebacks
- Configured at INFO level by default

Example:
```python
from hackathon.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("Processing document")
logger.error("Failed to process", exc_info=True)
```

### GPU Acceleration

Embeddings MUST use GPU (ROCm/CUDA) for reasonable performance. CPU-based embeddings are too slow for production use.

Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Progress Tracking

All processing steps use `rich.progress.track()` for visual feedback:
- Document chunking (per file)
- BM25 indexing (per chunk)
- Embedding generation (per batch)
- Contextual processing (per chunk)

### Error Handling

- Docling metadata uses Pydantic models, not dicts: use `getattr(meta, "field", default)` not `meta.get("field")`
- LLM responses can be inconsistent: use JSON mode + post-processing validation
- PostgreSQL unique constraints: ensure `node_path` is unique per document

## Common Issues

### Same Results for All Queries
This happens when the corpus is too small (< 10 documents). The system needs enough documents to differentiate between queries.

### Code Blocks/Lists Cut Off
Ensure you're using semantic block reassembly via `ContextExpander.get_semantic_block_text()`. This is already implemented in `cli/query.py`.

### Contextual Summaries with Prefixes
The LLM sometimes adds "Context:" or "Summary:" prefixes. These are stripped using `str.removeprefix()` in `contextual.py`.

### Database Rebuild Required
Any changes to chunking logic require rebuilding the database (`uv run clean` then `uv run process`).

## Development Workflow

1. Make code changes
2. Run `make lint` to check formatting
3. Run `make types` to check types
4. Run `make tests` to verify tests pass
5. Optionally run `make complexity` to check code complexity metrics
6. If database schema changed, run `uv run clean` then `uv run process`
7. Test queries with `uv run query`

## Libraries Used to Reduce Custom Code

The project leverages established Python libraries to minimize custom code:

- **python-frontmatter**: YAML frontmatter parsing (replaced ~60 lines of custom regex)
- **rich.progress**: Progress tracking with visual feedback (replaced tqdm)
- **functools.lru_cache**: Settings singleton pattern (built-in Python)
- **contextlib.contextmanager**: Database session management (built-in Python)
- **str.removeprefix()**: String prefix removal (Python 3.9+ built-in)
- **radon**: Code complexity analysis (cyclomatic complexity + maintainability index)

## References

- [Docling Documentation](https://github.com/DS4SD/docling)
- [pgvector Python Library](https://github.com/pgvector/pgvector-python)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
- [Granite Embedding Models](https://huggingface.co/ibm-granite/granite-embedding-30m-english)
- [python-frontmatter](https://github.com/eyeseast/python-frontmatter)
- [Radon Code Metrics](https://radon.readthedocs.io/)
