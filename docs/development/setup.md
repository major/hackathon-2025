# Development Setup

This guide walks you through setting up a development environment for the RAG system.

## Prerequisites ✅

### Required Software

- **Python 3.12+** 🐍
- **PostgreSQL 14+** 🐘 with pgvector extension
- **uv** 📦 (Python package manager)
- **Git** 🌿
- **CUDA/ROCm** 🎮 (optional but strongly recommended for GPU acceleration)

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB+ |
| Disk | 5 GB | 20 GB+ |
| GPU | None (CPU fallback) | NVIDIA/AMD with 4GB+ VRAM |

---

## Step 1: Install PostgreSQL with pgvector 🐘

### Ubuntu/Debian

```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Install build dependencies for pgvector
sudo apt install git build-essential postgresql-server-dev-all

# Clone and install pgvector
cd /tmp
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### macOS (Homebrew)

```bash
# Install PostgreSQL
brew install postgresql@14

# Start PostgreSQL service
brew services start postgresql@14

# Install pgvector
cd /tmp
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
make install
```

### Verify Installation

```bash
# Connect to PostgreSQL
psql -U postgres

# Create extension
CREATE EXTENSION vector;

# Check version
SELECT extversion FROM pg_extension WHERE extname = 'vector';
# Should show: 0.7.0 or higher

\q
```

---

## Step 2: Create Database 🗄️

```bash
# Connect as postgres user
psql -U postgres

# Create database
CREATE DATABASE rag_system;

# Connect to new database
\c rag_system

# Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

# Create user (optional)
CREATE USER rag_user WITH PASSWORD 'your_password_here';
GRANT ALL PRIVILEGES ON DATABASE rag_system TO rag_user;

\q
```

---

## Step 3: Install uv Package Manager 📦

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

---

## Step 4: Clone Repository 🌿

```bash
git clone https://github.com/yourusername/rag-system.git
cd rag-system
```

---

## Step 5: Install Dependencies 🔧

```bash
# Install all dependencies (including dev dependencies)
make install

# Or manually with uv
uv sync --dev
```

This installs:
- **Core**: SQLAlchemy, Pydantic, sentence-transformers, Docling
- **Database**: psycopg2-binary, pgvector
- **ML/AI**: anthropic, torch, rank-bm25
- **Dev tools**: pytest, ruff, pyright, mkdocs

---

## Step 6: Configure Environment Variables 🔐

Create `.env` file in project root:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_system

# Anthropic API (for contextual summaries)
ANTHROPIC_API_KEY=sk-ant-api03-...

# Embedding Model
EMBEDDING_MODEL=ibm-granite/granite-embedding-30m-english

# Document Processing
MARKDOWN_DIR=./blog/content/posts
MARKDOWN_PATTERN=**/*.md
MAX_CHUNK_SIZE=512
```

### Get Anthropic API Key

1. Sign up at https://console.anthropic.com/
2. Navigate to API Keys
3. Create new key
4. Copy to `.env` file

---

## Step 7: Verify GPU Support (Optional) 🎮

```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Expected output (with GPU):
# CUDA available: True

# Check GPU details
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Expected output:
# NVIDIA GeForce RTX 3080
```

**No GPU?** 😅 The system will fall back to CPU, but embeddings will be ~10-50x slower.

---

## Step 8: Initialize Database Schema 🏗️

```bash
# Run database initialization
uv run python -c "from hackathon.database import init_db; init_db()"
```

This creates all tables:
- `documents`
- `document_nodes`
- `embeddings`
- `bm25_index`
- `contextual_chunks`

Verify:

```bash
psql -U postgres -d rag_system -c "\dt"
```

Expected output:
```
              List of relations
 Schema |        Name         | Type  |  Owner
--------+---------------------+-------+----------
 public | bm25_index          | table | postgres
 public | contextual_chunks   | table | postgres
 public | document_nodes      | table | postgres
 public | documents           | table | postgres
 public | embeddings          | table | postgres
```

---

## Step 9: Add Sample Documents 📚

```bash
# Create blog directory structure
mkdir -p blog/content/posts

# Add a sample markdown file
cat > blog/content/posts/sample.md << 'EOF'
---
title: "Database Connection Pooling"
date: 2024-01-15
tags: ["postgresql", "performance"]
---

# Database Connection Pooling

Connection pooling reduces overhead by reusing database connections.

## Benefits

- Faster connection reuse
- Reduced latency
- Better resource management

## Configuration

```python
from psycopg2.pool import ThreadedConnectionPool

pool = ThreadedConnectionPool(
    minconn=1,
    maxconn=10,
    dsn="postgresql://localhost/mydb"
)
```

Set `pool_size` to 10 for optimal performance.
EOF
```

---

## Step 10: Process Documents 🔄

```bash
# Run the ingestion pipeline
uv run process
```

Expected output:
```
Starting document processing...

Step 1: Processing markdown files
Processing documents: 100%|████████████| 1/1 [00:02<00:00,  2.15s/it]

Step 2: Building BM25 index
Indexing chunks: 100%|█████████████████| 5/5 [00:00<00:00, 125.42it/s]

Step 3: Generating embeddings
Embedding batches: 100%|██████████████| 1/1 [00:03<00:00,  3.21s/it]

Step 4: Generating contextual summaries
Contextual processing (doc 1): 100%|██| 5/5 [00:08<00:00,  1.62s/it]

Processing complete!
Processed 1 documents with 5 chunks
```

---

## Step 11: Test Query System 🔍

```bash
# Interactive query
uv run query
```

```
🔍 RAG System Query Interface

Enter your query (or 'quit' to exit): database pooling

Searching for: database pooling

┌──────┬────────┬──────────┬────────────┬─────────────────────────────────┐
│ Rank │ Score  │ Type     │ Path       │ Preview                         │
├──────┼────────┼──────────┼────────────┼─────────────────────────────────┤
│ 1    │ 0.9234 │ paragraph│ chunk_0    │ Connection pooling reduces...   │
│ 2    │ 0.8421 │ code     │ chunk_3    │ pool = ThreadedConnectionPo...  │
│ 3    │ 0.7156 │ list_item│ chunk_1    │ Faster connection reuse         │
└──────┴────────┴──────────┴────────────┴─────────────────────────────────┘

Enter your query (or 'quit' to exit): quit

Goodbye! 👋
```

Success! 🎉

---

## Step 12: Run Tests ✅

```bash
# Run full test suite
make tests
```

Expected output:
```
Running pytest...
========================= test session starts ==========================
collected 9 items

tests/test_bm25.py::test_tokenize_text PASSED                     [ 11%]
tests/test_database.py::test_create_document PASSED               [ 22%]
tests/test_database.py::test_create_document_node PASSED          [ 33%]
...

========================= 9 passed in 2.45s ============================
```

---

## Development Tools 🛠️

### Code Quality

```bash
# Run linter
make lint

# Run type checker
make types

# Run complexity analysis
make complexity

# Run all checks
make all
```

### Documentation

```bash
# View docs locally with live reload
make docs-serve

# Build static docs
make docs-build
```

### Database Management

```bash
# Reset database (⚠️ deletes all data!)
uv run clean

# Rebuild from scratch
uv run clean && uv run process
```

---

## IDE Setup 💻

### VS Code

**Recommended extensions**:
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Ruff (charliermarsh.ruff)

**Settings** (`.vscode/settings.json`):
```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "ruff",
  "python.languageServer": "Pylance",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

### PyCharm

1. **Configure Python interpreter**:
   - Settings → Project → Python Interpreter
   - Add interpreter → Existing environment
   - Select: `.venv/bin/python`

2. **Enable type checking**:
   - Settings → Editor → Inspections
   - Enable: "Type checker" (Pyright)

3. **Configure code style**:
   - Settings → Tools → External Tools
   - Add Ruff formatter

---

## Troubleshooting 🔧

### Issue: `psycopg2` installation fails

**Error**: `Error: pg_config executable not found`

**Solution**:
```bash
# Ubuntu/Debian
sudo apt install libpq-dev

# macOS
brew install postgresql
```

### Issue: pgvector extension not found

**Error**: `ERROR:  extension "vector" is not available`

**Solution**: Reinstall pgvector (see Step 1)

### Issue: CUDA out of memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Reduce batch size in embedder.py
# Edit src/hackathon/processing/embedder.py
self.batch_size = 8  # Reduce from 32
```

Or use CPU:
```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
```

### Issue: Anthropic API rate limit

**Error**: `RateLimitError: 429 Too Many Requests`

**Solution**: Add delays between contextual summary generation:
```python
# Edit src/hackathon/processing/contextual.py
import time

# After each LLM call
time.sleep(1)  # Wait 1 second
```

### Issue: Database connection refused

**Error**: `psycopg2.OperationalError: could not connect to server`

**Solution**:
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Start if not running
sudo systemctl start postgresql

# Check connection
psql -U postgres -d rag_system -c "SELECT 1;"
```

---

## Next Steps 📚

- [Usage Guide](usage.md) - Learn how to use the system
- [Testing Guide](testing.md) - Write tests for new features
- [Contributing Guide](contributing.md) - Submit pull requests

---

## Quick Reference 📋

### Common Commands

```bash
# Development workflow
make install        # Install dependencies
make tests          # Run tests
make lint           # Check code style
make all            # Run all checks

# Document processing
uv run process      # Process all markdown files
uv run clean        # Reset database

# Querying
uv run query        # Interactive mode
uv run query "text" # Direct query

# Documentation
make docs-serve     # View docs locally
make docs-build     # Build static site
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://localhost/rag_system` |
| `ANTHROPIC_API_KEY` | Anthropic API key for LLM | Required |
| `EMBEDDING_MODEL` | HuggingFace model name | `ibm-granite/granite-embedding-30m-english` |
| `MARKDOWN_DIR` | Source documents directory | `./blog/content/posts` |
| `MARKDOWN_PATTERN` | Glob pattern for files | `**/*.md` |
| `MAX_CHUNK_SIZE` | Maximum chunk tokens | `512` |

### Database Connections

```python
# In Python code
from hackathon.database import get_db

db = next(get_db())
# ... use db
db.close()

# Or use context manager
from hackathon.database import session_scope

with session_scope() as db:
    # ... use db
    pass  # Auto-commits on success, rolls back on error
```

Ready to start developing! 🚀
