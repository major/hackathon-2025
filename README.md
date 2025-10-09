# RAG System with BM25, Semantic Search, and Contextual Retrieval

A proof-of-concept Retrieval-Augmented Generation (RAG) system built for processing markdown documents with hierarchical structure support.

## Features

- **Markdown Processing**: Parses markdown files with automatic YAML frontmatter removal
- **Hierarchical Document Structure**: Maintains document tree with parent-child relationships
- **Hybrid Search**:
  - BM25 keyword-based search
  - Semantic similarity using granite-embedding-30m-english
  - Combined ranking with adjustable weights
- **Contextual Retrieval**: Implements Anthropic's Contextual Retrieval approach for improved search accuracy
- **Context Expansion**: Walk up document tree to get broader context when needed
- **ROCm/CUDA Support**: GPU-accelerated embeddings using PyTorch with ROCm 6.3 (nightly)
- **Progress Tracking**: Granular progress bars for all processing steps

## Prerequisites

- Python 3.12+
- PostgreSQL with pgvector extension
- AMD GPU with ROCm 6.3 (or NVIDIA GPU with CUDA)
- OpenAI-compatible LLM API (e.g., granite4 via ramalama)

## Installation

```bash
# Install dependencies (includes ROCm PyTorch nightly from configured index)
uv sync --dev
```

The project is configured to use PyTorch nightly builds with ROCm 6.3 support via the custom index in `pyproject.toml`.

## Configuration

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Update `.env` with your configuration:

```env
# Database Configuration
DB_HOST=127.0.0.1
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=secrete
DB_NAME=hackathon

# OpenAI-compatible API (granite4 via ramalama)
LLM_API_BASE=http://127.0.0.1:8086/v1
LLM_MODEL=granite4
LLM_API_KEY=not-needed

# Embedding Configuration
EMBEDDING_MODEL=ibm-granite/granite-embedding-30m-english
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=32

# Processing Configuration
MARKDOWN_DIRECTORY=blog/content/posts
MARKDOWN_PATTERN=**/*.md
```

## Database Setup

Initialize PostgreSQL with pgvector:

```bash
# Create database
createdb hackathon

# Enable pgvector extension
psql hackathon -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

The application will automatically create the required tables on first run.

## Usage

### Process Documents

Process all markdown files in the configured directory:

```bash
uv run process
```

This will:
1. Parse and chunk markdown documents
2. Build hierarchical document tree
3. Create BM25 indices
4. Generate embeddings (GPU-accelerated)
5. Create contextual summaries for each chunk

### Query the System

Run interactive queries:

```bash
uv run query
```

Or provide a query directly:

```bash
uv run query "your search query here"
```

### Clean Database

Reset the database by dropping all tables:

```bash
uv run clean
```

This will prompt for confirmation before deleting all data.

## Development

### Run Tests

```bash
make tests
```

### Lint Code

```bash
make lint
```

### Type Checking

```bash
make types
```

### Run All Checks

```bash
make all
```

## Project Structure

```
src/hackathon/
├── config.py              # Configuration management
├── models/
│   ├── database.py        # SQLAlchemy models
│   └── schemas.py         # Pydantic schemas
├── database/
│   ├── connection.py      # Database connection
│   └── operations.py      # CRUD operations
├── processing/
│   ├── parser.py          # Markdown parsing
│   ├── markdown_chunker.py # Document chunking
│   ├── bm25.py            # BM25 indexing
│   ├── embedder.py        # Embedding generation
│   └── contextual.py      # Contextual retrieval
├── retrieval/
│   ├── search.py          # Hybrid search
│   └── context_expansion.py # Tree traversal
├── cli/
│   ├── process.py         # Processing command
│   └── query.py           # Query command
└── utils/
    └── progress.py        # Progress bars
```

## Architecture

### Data Flow

1. **Document Processing**:
   - Markdown files → Parser (removes frontmatter) → Chunker (creates hierarchy) → Database
   - Chunks → BM25 Tokenizer → BM25 Index
   - Chunks → Embedding Generator (ROCm/CUDA) → Vector Embeddings
   - Chunks + Context → LLM → Contextual Summaries

2. **Query Processing**:
   - Query → BM25 Search + Semantic Search → Hybrid Ranking
   - Results → Context Expansion (optional) → Enhanced Results

### Database Schema

- **documents**: Metadata about source files
- **document_nodes**: Hierarchical tree structure with parent-child relationships
- **embeddings**: Vector embeddings for semantic search (pgvector)
- **bm25_index**: Tokenized content for keyword search
- **contextual_chunks**: Contextualized chunks for improved retrieval

## Performance Notes

- GPU-accelerated embedding generation is **critical** for reasonable processing times
- Processing time depends on:
  - Number and size of documents
  - Embedding batch size (configurable)
  - GPU performance
  - LLM API response time for contextual summaries

## Troubleshooting

### ROCm Not Available

If you see "CUDA/ROCm is not available":
1. Verify ROCm installation: `rocm-smi`
2. Check PyTorch ROCm build: `python -c "import torch; print(torch.cuda.is_available())"`
3. Ensure dependencies installed correctly with `uv sync`

### Database Connection Errors

- Verify PostgreSQL is running: `systemctl status postgresql`
- Check credentials in `.env`
- Ensure pgvector extension is installed

### LLM API Errors

- Verify the OpenAI-compatible API is running
- Check `LLM_API_BASE` in `.env`
- Test with: `curl http://127.0.0.1:8086/v1/models`

## License

See LICENSE file for details.

## Credits

- Built with [Docling](https://github.com/DS4SD/docling) for document processing
- Uses [pgvector](https://github.com/pgvector/pgvector) for vector similarity search
- Implements [Anthropic's Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
- Embeddings from IBM's [Granite models](https://huggingface.co/ibm-granite)
