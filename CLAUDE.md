# Claude Code Guide - RAG System

This document provides context for Claude Code instances working in this repository.

## Quick Commands

```bash
# Document processing pipeline
uv run download-tokenizer           # Download BERT tokenizer for offline use (one-time setup)
uv run process                      # Process all markdown files with contextual retrieval (uses IBM Watsonx)
uv run process --skip-contextual    # Skip contextual summaries (faster, no LLM calls)
uv run query                        # Interactive query interface
uv run query "text"                 # Direct query (BM25 only)
uv run query "text" --rerank        # Query with IBM Watsonx semantic reranking (best quality!)
uv run query "text" --expand-query  # Query with semantic query expansion (improves recall!)
uv run query "text" --expand-query --rerank  # Combine query expansion + reranking (ultimate quality!)
uv run query "text" --neighbors 2   # Show 2 neighbors before/after matched chunk
uv run clean                        # Drop all database tables and start fresh

# Reranking analysis
uv run python scripts/compare_reranking.py "your query"  # Compare BM25 vs IBM Watsonx reranked results
uv run python scripts/benchmark_reranking.py             # Benchmark performance

# Debugging
uv run python scripts/docling_debug.py file.md        # See Docling's JSON output
uv run python scripts/docling_debug.py file.md out.json  # Save JSON to file

# Development
make tests              # Run pytest suite
make lint               # Run ruff linter and formatter
make types              # Run pyright type checker
make complexity         # Run radon code complexity analysis
make deadcode           # Find unused code with vulture (80% confidence)
make deadcode-aggressive # Find unused code (60% confidence, more results)
make all                # Run all checks (lint, types, tests)
make install            # Install dependencies with uv

# Documentation
make docs-serve       # Start MkDocs dev server with live reload (http://127.0.0.1:8000)
make docs-build       # Build MkDocs static site for production (outputs to site/)
```

## High-Level Architecture

This is a proof-of-concept RAG system using **BM25-only contextual retrieval** (no vector embeddings). It implements Anthropic's Contextual Retrieval pattern with pure keyword search, enhanced by IBM Watsonx AI for LLM-powered contextual summaries and semantic reranking.

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
    ├── Multi-field BM25 indexing (bm25s library):
    │   ├── full_text: Complete chunk content
    │   ├── headings: Hierarchical heading context
    │   ├── summary: First 1-2 sentences
    │   └── contextual_text: LLM-generated contextual summary + text (IBM Watsonx Granite)
    └── PostgreSQL FTS (tsvector): Full-text search with ts_rank
```

### Query Pipeline

```
User query
    ↓
Optional: Query expansion with IBM Watsonx 🔍
    └── Generate 3-5 semantic variations of the query
    └── Example: "configure logging" → ["set up logs", "enable logging", "log configuration"]
    ↓
Five-way Reciprocal Rank Fusion (RRF):
    ├── BM25 full_text search (per query variation)
    ├── BM25 headings search (per query variation)
    ├── BM25 summary search (per query variation)
    ├── BM25 contextual_text search (contextual retrieval, per query variation)
    └── PostgreSQL FTS (tsvector, per query variation)
    ↓
Combined ranking with RRF scores (top 50-100 candidates)
    ↓
Optional: IBM Watsonx semantic reranking 🎯
    └── Cross-encoder reranks candidates (returns top K)
    ↓
Context expansion (optional):
    └── Neighbor retrieval (previous/next chunks)
    ↓
Results with expanded context
```

### Core Components

**Processing (`src/hackathon/processing/`)**:
- `docling_processor.py`: Docling integration for document parsing and chunking
  - `extract_yaml_frontmatter()`: YAML parsing using python-frontmatter library
  - `process_document_with_docling()`: Creates temp file without frontmatter for Docling processing
- `bm25.py`: Multi-field BM25 indexing (bm25s library) with 4 fields + Reciprocal Rank Fusion
- `contextual.py`: LLM-based contextual summaries using IBM Watsonx Granite

**Retrieval (`src/hackathon/retrieval/`)**:
- `multifield_searcher.py`: `MultiFieldBM25Searcher` class with 5-way RRF (4 BM25 + PostgreSQL FTS) + optional query expansion + optional reranking
- `query_expansion.py`: `QueryExpander` class for generating semantic query variations using IBM Watsonx
- `reranker.py`: `WatsonxReranker` class for semantic reranking (optional, requires Watsonx credentials)
- `context_expansion.py`: Neighbor retrieval for expanded context

**Database (`src/hackathon/database/`)**:
- `connection.py`: SQLAlchemy setup with PostgreSQL FTS + `session_scope()` context manager
- `operations.py`: CRUD operations for documents, nodes, BM25 indexes, etc.

**Models (`src/hackathon/models/`)**:
- `database.py`: SQLAlchemy ORM models (Document, DocumentNode, MultiFieldBM25Index)
- `schemas.py`: Pydantic schemas for validation and API contracts

## Key Technical Details

### Database Schema

- **documents**: Source file metadata
- **document_nodes**: Flat chunk structure with sequential positions
  - `node_type`: From Docling (e.g., "paragraph", "list_item", "code", "section_header", "table")
  - `is_leaf`: Boolean flag for indexable chunks
  - `position`: Sequential position for neighbor lookup
  - `metadata`: JSON field with Docling metadata (headings, docling_types, origin)
  - `text_search`: PostgreSQL tsvector for full-text search
- **multifield_bm25_index**: Four searchable fields per chunk
  - `full_text`: Complete chunk content
  - `headings`: Hierarchical heading context
  - `summary`: First 1-2 sentences
  - `contextual_text`: LLM-generated contextual summary + original text

### Docling Integration

Docling provides:
- Automatic document structure detection (headings, lists, code blocks, tables)
- HybridChunker for intelligent chunking that respects semantic boundaries
- Rich metadata per chunk:
  - `doc_items`: List of content types in this chunk
  - `headings`: Hierarchical heading context (e.g., "System Logging, Using journalctl")
  - `origin`: Document elements this chunk came from

Important: YAML frontmatter is extracted using the python-frontmatter library. The clean content (without frontmatter) is written to a temporary file that Docling processes, ensuring frontmatter doesn't interfere with document structure detection.

### HuggingFace Offline Mode 🤗

The system runs in **offline mode** to avoid network calls during processing:

**First-Time Setup:**
```bash
# Download the BERT tokenizer once (required for offline mode)
uv run download-tokenizer
```

This downloads `bert-base-uncased` (~500MB) to `~/.cache/huggingface/`. After this one-time download, all processing runs offline!

**How it works:**
- Environment variables force offline mode: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`
- Set at module level in `docling_processor.py` (before any imports)
- No network calls during parallel processing (fast & reliable!)
- Tokenizer is used for accurate chunking (respects semantic boundaries)

**Parallel Processing Benefits:** 🚀
- Document processing uses `ThreadPoolExecutor` (4 concurrent threads)
- Contextual summaries use `ThreadPoolExecutor` (10 concurrent LLM API calls)
- Each thread creates its own database session (SQLAlchemy sessions are not thread-safe)
- Expected speedup: ~4x for documents, ~10x for contextual summaries!

### Multi-field BM25 Indexing 📚

Uses the `bm25s` library to build four separate BM25 indexes per chunk, stored as persistent files:
1. **full_text**: Complete chunk content for primary retrieval
2. **headings**: Hierarchical context (e.g., "Installation > Dependencies > npm")
3. **summary**: First 1-2 sentences for topic matching
4. **contextual_text**: LLM-enhanced text for improved relevance

The indexes are saved to disk (default: `.bm25s_index/`) and loaded at query time for fast retrieval.

### Reciprocal Rank Fusion (RRF) 🔀

RRF combines rankings from five different search methods:
- **BM25 fields (4x)**: full_text, headings, summary, contextual_text
- **PostgreSQL FTS (1x)**: tsvector with ts_rank scoring

Formula: `score = sum(1 / (k + rank))` where k=60 (standard from literature)

This approach outperforms weighted averaging and doesn't require tuning weights!

### Contextual Retrieval 🤖

Implements Anthropic's Contextual Retrieval pattern using **IBM Watsonx Granite**:
1. For each chunk, generate a contextual summary explaining what it's about and how it relates to the document
2. Combine summary + original text into `contextual_text` field
3. Index the enriched text in BM25 for improved retrieval accuracy

Example transformation:
```
Original chunk: "Run npm install to download dependencies."

Contextual summary: "This section explains how to install project dependencies using npm."

Indexed text: "This section explains how to install project dependencies using npm. Run npm install to download dependencies."
```

The LLM adds situational context that improves keyword matching, especially for queries that don't use the exact terms in the original text.

**Important**: Uses `str.removeprefix()` to strip common prefixes ("Context:", "Summary:") that LLMs sometimes add despite instructions. ✂️

### IBM Watsonx Configuration 🔑

The system requires IBM Watsonx credentials for contextual summary generation and reranking:

```bash
export WATSONX_API_KEY="your-api-key-here"
export WATSONX_PROJECT_ID="your-project-id-here"
```

Or add to `.env` file:
```
WATSONX_API_KEY=your-api-key-here
WATSONX_PROJECT_ID=your-project-id-here
WATSONX_URL=https://us-south.ml.cloud.ibm.com

# Optional: Override default models
WATSONX_LLM_MODEL=ibm/granite-4-h-small
WATSONX_RERANKER_MODEL=cross-encoder/ms-marco-minilm-l-12-v2
```

Get your API key at: https://cloud.ibm.com/iam/apikeys

**Skip contextual generation** with `--skip-contextual` flag if you don't want to use the API (faster processing, but no contextual enhancement).

### Semantic Reranking 🎯 (Optional)

For even better relevance, the system supports **IBM Watsonx reranking** - a two-stage retrieval approach using the native IBM Watsonx AI SDK (no LangChain!):

**Stage 1: Fast Retrieval** (BM25 + RRF)
- Retrieve top 50-100 candidates using existing 5-way RRF
- Fast keyword/lexical matching
- Great recall (finds relevant documents)

**Stage 2: Semantic Reranking** (IBM Watsonx)
- Cross-encoder model scores query-document relevance
- Semantic understanding ("configure" matches "setup")
- Intent detection ("how do I...?" prioritizes instructions)
- Reorders candidates and returns top N

**Usage:**
```bash
# BM25 only (fast)
uv run query "configure logging"

# With reranking (best quality!)
uv run query "configure logging" --rerank

# Adjust candidate count (default: 50)
uv run query "configure logging" --rerank --candidates 100
```

**Configuration:**
Uses the same Watsonx credentials as contextual retrieval (see above).

**Benefits:**
- 🏢 **Enterprise-grade:** Production-ready semantic reranking
- 🔒 **Data privacy:** Secure, compliant infrastructure
- 📊 **Reliability:** Consistent performance and uptime
- 🎯 **Quality:** Optimized for business use cases

**When to Use Reranking:**
- ✅ Conceptual queries ("best practices for X")
- ✅ Intent-based searches ("how do I...?", "why does...?")
- ✅ Semantic matching needed ("configure" vs "set up")
- ✅ Natural language questions (3+ words)
- ❌ Short keyword searches (1-2 words like "pgadmin", "docker")
- ❌ Exact technical term matching (BM25 is better for keywords)
- ❌ Speed-critical applications (<300ms required)

**⚠️ Important Note:** The reranker may reduce precision for very short keyword queries (1-2 words). For exact keyword matches like "pgadmin" or "kubernetes", use BM25-only (without `--rerank`). The system will warn you when this happens.

**Implementation:** See `src/hackathon/retrieval/reranker.py` for the `WatsonxReranker` class.

### Query Expansion 🔍 (Optional)

For even better recall (finding more relevant documents), the system supports **query expansion** - generating semantic variations of your query using IBM Watsonx:

**How It Works:**
1. Original query: "configure logging"
2. LLM generates variations: ["set up logs", "enable logging", "log configuration"]
3. System searches with ALL variations (original + generated)
4. Results merged using RRF (Reciprocal Rank Fusion)

**Why Use Query Expansion?**
- 🎯 **Better recall:** Finds documents that use different terminology
- 📚 **Vocabulary mismatch:** Solves the problem when users and documents use different words for the same concept
- 🔄 **Synonym matching:** "configure" → "set up", "setup", "configure"
- 💡 **Concept expansion:** "logging" → "logs", "log files", "system logging"

**Usage:**
```bash
# BM25 only (fast)
uv run query "configure logging"

# With query expansion (better recall!)
uv run query "configure logging" --expand-query

# Adjust number of variations (default: 3)
uv run query "configure logging" --expand-query --query-variations 5

# Combine with reranking for ultimate quality!
uv run query "configure logging" --expand-query --rerank
```

**Configuration:**
Uses the same Watsonx credentials as contextual retrieval and reranking (see above).

**Benefits:**
- 🎯 **20-30% improvement in recall** for natural language queries
- 🔍 **Finds documents missed by exact keyword matching**
- 🚀 **Fast:** Query expansion happens in parallel with search
- 🤖 **Smart:** LLM understands context and generates relevant variations

**When to Use Query Expansion:**
- ✅ Natural language questions ("how do I configure X?")
- ✅ Conceptual queries ("best practices for logging")
- ✅ When you're not sure of exact terminology
- ✅ When searching across documents with varied writing styles
- ❌ Exact keyword searches (already working well)
- ❌ Very short queries (1-2 words - expansion may add noise)
- ❌ Speed-critical applications (adds ~200-500ms LLM latency)

**How Query Expansion Differs from Reranking:**
- **Query Expansion (Stage 1):** Improves **recall** by finding MORE relevant documents through semantic query variations
- **Reranking (Stage 2):** Improves **precision** by reordering results for better relevance

**Best Results:** Use both together! 🚀
```bash
uv run query "how do I set up logging?" --expand-query --rerank
```

This gives you:
1. Better recall from query expansion (finds more relevant docs)
2. Better precision from reranking (ranks them correctly)
3. Ultimate retrieval quality! 🎯

**Performance:**
- Query expansion adds ~200-500ms (LLM call to generate variations)
- Can search with multiple variations in parallel (fast!)
- Total impact: ~300-700ms for typical queries with 3 variations

**Implementation:** See `src/hackathon/retrieval/query_expansion.py` for the `QueryExpander` class.

## Important Patterns

### ⛔ Framework Ban Policy

**BANNED FRAMEWORKS:** This project does NOT use the following frameworks:
- ❌ **LangChain** - No LangChain imports or dependencies
- ❌ **LlamaIndex** - No LlamaIndex imports or dependencies
- ❌ **Haystack** - No Haystack imports or dependencies

**Why?** We prefer direct SDK usage for:
- 🎯 **Full control** over implementation details
- 📦 **Minimal dependencies** and smaller attack surface
- 🔍 **Transparency** - know exactly what's happening
- 🚀 **Performance** - no framework overhead

Always use official SDKs directly (IBM Watsonx AI SDK, etc.) instead of framework wrappers.

### Pydantic Settings

Configuration is managed via `src/hackathon/config.py` using pydantic-settings:
- Loads from `.env` file
- Type validation
- Singleton pattern via `@lru_cache`

### Centralized Logging 📢

All logging is configured centrally in `src/hackathon/__init__.py` on package import! 🎯

**What's Configured:**
- 🔇 **Global silence**: gRPC, transformers, bm25s all set to CRITICAL
- ⚠️ **App logging**: WARNING level (Rich formatting with tracebacks)
- 🤫 **Environment vars**: GRPC_VERBOSITY, GLOG_minloglevel, etc.
- 🚫 **Warnings filter**: All library warnings suppressed

**Usage in your code:**
```python
from hackathon.utils.logging import get_logger

logger = get_logger(__name__)
logger.warning("Something important!")  # Will show
logger.info("Processing document")     # Won't show (below WARNING)
logger.error("Failed!", exc_info=True) # Will show with traceback
```

**No need to configure logging in individual files!** It's all handled on import! ✨

### Emoji Usage 😊

**IMPORTANT**: This codebase LOVES emojis! Use them liberally for:
- Documentation section headers (e.g., 📚 🔀 🤖 🔑)
- Log messages and console output for visual clarity
- Code comments for humor and helpfulness
- Commit messages (when appropriate)

Emojis make code more readable and fun! Don't be shy - embrace the emoji life! 🎉

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

## Performance Notes ⚡

### Query Startup Time (~0.5 seconds)

Query startup is now **blazingly fast** thanks to lazy imports! 🚀

**Startup breakdown:**
- Package initialization: ~0.24s (one-time Python overhead)
- BM25 index loading: ~0.27s (loads 4 indexes from disk)
- **Total: ~0.5s** for the first query

**Key optimization - Lazy Imports:**
Docling is only imported during document processing (in `_convert_document_with_docling()` and `_chunk_document()`), not during query operations. This avoids a **2.3 second startup penalty** when running queries.

- ❌ **Before:** Docling imported at module level → 2.8s startup
- ✅ **After:** Docling imported on first use → 0.5s startup (5.6x faster!)

**BM25 Index Loading:**
The first query loads **4 BM25 indexes** from disk into memory:
- `full_text` index
- `headings` index
- `summaries` index
- `contextual_text` index

This is a **one-time cost** per query session (~0.27s). After loading:
- ✅ Subsequent queries are **instant** (already in memory)
- ✅ The indexes use `mmap=True` for efficient memory usage
- ✅ Total index size depends on your corpus

**Does lazy loading slow down document processing?**
No! Python caches imported modules, so:
- First document processed: 2.3s Docling import + processing time
- Subsequent documents: 0s import + processing time (cached)

The 2.3s import cost is amortized across all documents in a processing session, just like before.

**Tip for even faster queries:**
Use interactive mode: `uv run query` (load indexes once, query many times)

## Common Issues

### Same Results for All Queries
This happens when the corpus is too small (< 10 documents). The system needs enough documents to differentiate between queries.

### Code Blocks/Lists Cut Off
Ensure you're using semantic block reassembly via `ContextExpander.get_semantic_block_text()`. This is already implemented in `cli/query.py`.

### Contextual Summaries with Prefixes ✂️
The LLM sometimes adds "Context:" or "Summary:" prefixes despite instructions. These are stripped using `str.removeprefix()` in `contextual.py`.

### Missing API Key 🔑
If you get errors about `WATSONX_API_KEY`, you need to set your Watsonx API key in `.env` or use `--skip-contextual` to disable contextual retrieval.

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

## References 📖

- [Docling Documentation](https://github.com/DS4SD/docling) - Document parsing and chunking
- [bm25s Library](https://github.com/xhluca/bm25s) - Fast BM25 implementation
- [Anthropic Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) - Original pattern
- [Anthropic Contextual Retrieval Cookbook](https://github.com/anthropics/claude-cookbooks/blob/main/skills/contextual-embeddings/guide.ipynb) - Implementation guide
- [IBM Watsonx AI](https://www.ibm.com/watsonx) - Enterprise AI platform for LLM inference and reranking
- [IBM Watsonx AI Python SDK](https://ibm.github.io/watsonx-ai-python-sdk/) - Python SDK documentation
- [python-frontmatter](https://github.com/eyeseast/python-frontmatter) - YAML parsing
- [Radon Code Metrics](https://radon.readthedocs.io/) - Code complexity analysis
