# RAG System Developer Documentation

Welcome to the developer documentation for our hybrid Retrieval-Augmented Generation (RAG) system. This system combines multiple search strategies to provide accurate and context-rich document retrieval.

## What is This System?

This is a proof-of-concept RAG system that processes markdown documents and enables intelligent search through a combination of:

- **BM25 keyword search** (traditional full-text search)
- **Semantic vector search** (meaning-based similarity)
- **Contextual retrieval** (LLM-enhanced context)
- **Hierarchical context expansion** (document structure awareness)

## Quick Start

```bash
# Install dependencies
make install

# Process documents
uv run process

# Query the system
uv run query "your search query here"

# Clean and rebuild
uv run clean && uv run process
```

## System Architecture at a Glance

```
┌─────────────┐
│  Markdown   │
│   Files     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│   Docling Processing Pipeline   │
│  (Parse → Chunk → Extract Meta) │
└──────────────┬──────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│           PostgreSQL Database                │
│  ┌────────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Documents  │  │  Nodes   │  │  B
M25   │ │
│  │ & Metadata │  │ (chunks) │  │  Index   │ │
│  └────────────┘  └──────────┘  └──────────┘ │
│  ┌────────────┐  ┌──────────┐               │
│  │ Embeddings │  │Contextual│               │
│  │ (pgvector) │  │ Chunks   │               │
│  └────────────┘  └──────────┘               │
└───────────────────┬──────────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Hybrid Search      │
         │  (BM25 + Semantic)  │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │ Context Expansion   │
         │ & Result Assembly   │
         └─────────────────────┘
```

## Key Technologies

### Core Stack
- **Python 3.9+** - Primary language
- **PostgreSQL** - Database with pgvector extension
- **SQLAlchemy** - ORM and database operations
- **Pydantic** - Data validation and settings

### AI/ML Components
- **Docling** - Document parsing and intelligent chunking
- **sentence-transformers** - Embedding generation (granite-embedding-30m-english)
- **rank-bm25** - BM25 scoring implementation
- **Anthropic Claude** - Contextual summary generation

### Supporting Libraries
- **python-frontmatter** - YAML metadata parsing
- **rich** - Terminal UI and progress tracking
- **pgvector** - Vector similarity search in PostgreSQL

## Documentation Structure

### For New Developers
1. Start with [Architecture Overview](architecture/overview.md)
2. Understand the [Data Flow](architecture/data-flow.md)
3. Review [AI/ML Concepts](concepts/rag.md)

### For Implementation
1. [Ingestion Pipeline](components/ingestion.md) - How documents are processed
2. [Search & Retrieval](components/search.md) - How queries work
3. [Database Schema](architecture/database.md) - Data model details

### For Contributing
1. [Development Setup](development/setup.md)
2. [Testing Guide](development/testing.md)
3. [Contributing Guidelines](development/contributing.md)

## Design Philosophy

This system prioritizes:

1. **Accuracy over speed** - Multiple search strategies ensure relevant results
2. **Context preservation** - Hierarchical structure and semantic blocks
3. **Developer experience** - Clear code structure, comprehensive logging
4. **Maintainability** - All complexity ratings ≤ B, modular design

## Common Use Cases

### Search for Technical Concepts
```python
from hackathon.database import get_db
from hackathon.retrieval import HybridSearcher

db = next(get_db())
searcher = HybridSearcher(db)
results = searcher.hybrid_search("How to configure logging?", top_k=5)
```

### Expand Context for Results
```python
from hackathon.retrieval import ContextExpander

expander = ContextExpander(db)
full_context = expander.build_context_text(node, depth=2)
```

### Process New Documents
```bash
# Add markdown files to configured directory
uv run process
```

## Performance Characteristics

- **Corpus size**: Optimized for ~1,000-10,000 documents
- **Query latency**: ~100-500ms for hybrid search
- **Indexing speed**: ~1-5 docs/second (includes LLM calls)
- **GPU acceleration**: Required for embedding generation

## Next Steps

- Explore the [Architecture Overview](architecture/overview.md) to understand system design
- Review [AI/ML Concepts](concepts/rag.md) for background on RAG systems
- Check [Development Setup](development/setup.md) to start contributing
