# Usage Guide

This guide covers common tasks and workflows for using the RAG system effectively.

## Document Processing 📝

### Processing New Documents

```bash
# Process all markdown files in configured directory
uv run process
```

The system will:
1. ✅ Extract YAML frontmatter
2. ✅ Parse document structure with Docling
3. ✅ Create intelligent chunks
4. ✅ Build BM25 index
5. ✅ Generate vector embeddings
6. ✅ Create contextual summaries

### Incremental Updates

Currently, the system processes ALL documents on each run. To update a single document:

```bash
# Option 1: Clean and rebuild (⚠️ deletes everything)
uv run clean
uv run process

# Option 2: Delete specific document in database (future feature)
# Coming soon!
```

### Monitoring Progress

Watch for progress bars showing:
- 📄 Document parsing
- 🔢 BM25 indexing
- 🧠 Embedding generation
- 💬 Contextual summary creation

```
Step 1: Processing markdown files
Processing documents: 100%|████████████| 25/25 [00:45<00:00,  1.80s/it]
```

---

## Querying the System 🔍

### Interactive Mode

```bash
uv run query
```

**Features**:
- Type queries naturally
- See top-5 results in a table
- Automatic context expansion for top result
- Type `quit` to exit

**Example session**:
```
🔍 RAG System Query Interface

Enter your query (or 'quit' to exit): how to configure database pooling

Searching for: how to configure database pooling

┌──────┬────────┬──────────┬────────────┬────────────────────────────────┐
│ Rank │ Score  │ Type     │ Path       │ Preview                        │
├──────┼────────┼──────────┼────────────┼────────────────────────────────┤
│ 1    │ 0.9234 │ code     │ chunk_42   │ pool = ThreadedConnectionPo... │
│ 2    │ 0.8421 │ paragraph│ chunk_15   │ Configure pool_size to 10...   │
│ 3    │ 0.7156 │ list_item│ chunk_23   │ Set pool_pre_ping=True...      │
└──────┴────────┴──────────┴────────────┴────────────────────────────────┘

╭──────────────────── Full Context (with Semantic Block) ─────────────────────╮
│ [Semantic Block]                                                            │
│ from psycopg2.pool import ThreadedConnectionPool                            │
│                                                                              │
│ pool = ThreadedConnectionPool(                                              │
│     minconn=1,                                                              │
│     maxconn=10,                                                             │
│     pool_size=10,                                                           │
│     dsn="postgresql://localhost/mydb"                                       │
│ )                                                                            │
│                                                                              │
│ [Hierarchical Context]                                                      │
│ [section_header] PostgreSQL Guide > Connection Pooling                      │
╰──────────────────────────────────────────────────────────────────────────────╯

Enter your query (or 'quit' to exit): quit

Goodbye! 👋
```

### Direct Query

```bash
# Single query, then exit
uv run query "postgres connection pooling"
```

Useful for:
- Scripting
- Testing specific queries
- CI/CD integration

### Programmatic Usage

**In Python scripts**:

```python
from hackathon.database import get_db
from hackathon.retrieval import HybridSearcher

# Get database session
db = next(get_db())

# Create searcher
searcher = HybridSearcher(db)

# Perform search
results = searcher.hybrid_search("postgres pooling", top_k=5)

# Print results
for idx, result in enumerate(results, 1):
    print(f"{idx}. [{result.score:.4f}] {result.text_content[:100]}...")

db.close()
```

---

## Search Strategies 🎯

### Keyword-Heavy Queries

For exact technical terms, favor BM25:

```python
results = searcher.hybrid_search(
    "pool_size configuration parameter",
    bm25_weight=0.6,  # Increase BM25 weight
    semantic_weight=0.4
)
```

**Best for**:
- Configuration parameters (`pool_size`, `max_connections`)
- Function names (`ThreadedConnectionPool`)
- Error codes (`ERRNO 2006`)

### Conceptual Queries

For understanding topics, favor semantic:

```python
results = searcher.hybrid_search(
    "how to improve database performance",
    bm25_weight=0.3,
    semantic_weight=0.7  # Increase semantic weight
)
```

**Best for**:
- "How to..." questions
- Troubleshooting ("connection keeps dropping")
- General concepts ("best practices for caching")

### Balanced Queries

Use default weights (40% BM25 / 60% semantic):

```python
results = searcher.hybrid_search("postgres pool configuration")
# Uses default: bm25_weight=0.4, semantic_weight=0.6
```

**Best for**:
- Mixed technical + conceptual queries
- Unknown query type
- General search

---

## Context Expansion 🌳

### Basic Context

Show hierarchical context only:

```python
from hackathon.retrieval import ContextExpander

expander = ContextExpander(db)

# Get top result
result = results[0]
node = db.query(DocumentNode).get(result.node_id)

# Build context (2 levels of ancestors)
context = expander.build_context_text(node, depth=2)
print(context)
```

**Output**:
```
[section_header] Database Guide > Connection Pooling

[paragraph] Connection pooling reduces overhead...

[current: code] pool = ThreadedConnectionPool(...)
```

### With Siblings

Include sibling nodes for more context:

```python
context = expander.build_context_text(
    node,
    depth=2,
    include_siblings=True
)
```

**Output**:
```
[section_header] Database Guide > Connection Pooling

[sibling: paragraph] Benefits of pooling...

[sibling: list] - Faster connections...

[current: code] pool = ThreadedConnectionPool(...)
```

### Semantic Block Reassembly

For code/list/table chunks:

```python
# Try to reassemble semantic blocks
semantic_block = expander.get_semantic_block_text(node)

if semantic_block:
    print("[Complete Code Block]")
    print(semantic_block)
else:
    print("[Single Chunk]")
    print(node.text_content)
```

### Combined Approach

Best practice - use both:

```python
# Try semantic block reassembly
semantic_block = expander.get_semantic_block_text(node)
has_block = semantic_block and semantic_block != node.text_content

# Build hierarchical context (exclude current if showing semantic block)
context = expander.build_context_text(
    node,
    depth=2,
    exclude_current=has_block
)

# Display
if has_block:
    print("[Semantic Block]")
    print(semantic_block)
    print("\n[Hierarchical Context]")
    print(context)
else:
    print(context)
```

---

## Database Management 🗄️

### Inspecting Documents

```sql
-- List all documents
psql -U postgres -d rag_system -c "
SELECT id, filename, created_at
FROM documents
ORDER BY created_at DESC;
"

-- Count chunks per document
psql -U postgres -d rag_system -c "
SELECT d.filename, COUNT(dn.id) as chunks
FROM documents d
LEFT JOIN document_nodes dn ON d.id = dn.document_id
WHERE dn.is_leaf = TRUE
GROUP BY d.id, d.filename;
"
```

### Checking Index Coverage

```sql
-- Check indexing status
psql -U postgres -d rag_system -c "
SELECT
    COUNT(DISTINCT dn.id) as total_chunks,
    COUNT(DISTINCT e.node_id) as embedded,
    COUNT(DISTINCT b.node_id) as bm25_indexed,
    COUNT(DISTINCT c.node_id) as contextual
FROM document_nodes dn
LEFT JOIN embeddings e ON dn.id = e.node_id
LEFT JOIN (SELECT DISTINCT node_id FROM bm25_index) b ON dn.id = b.node_id
LEFT JOIN contextual_chunks c ON dn.id = c.node_id
WHERE dn.is_leaf = TRUE;
"
```

**Expected**: All counts should match for fully indexed corpus.

### Viewing Chunk Metadata

```sql
-- Inspect chunk details
psql -U postgres -d rag_system -c "
SELECT
    id,
    node_type,
    metadata->>'headings' as headings,
    metadata->>'docling_types' as types,
    LEFT(text_content, 80) as preview
FROM document_nodes
WHERE document_id = 1 AND is_leaf = TRUE
ORDER BY id
LIMIT 10;
"
```

### Clearing Database

```bash
# Drop all tables and rebuild schema
uv run clean

# Verify tables are gone
psql -U postgres -d rag_system -c "\dt"
```

---

## Performance Tuning ⚡

### Batch Size for Embeddings

Edit `src/hackathon/processing/embedder.py`:

```python
class EmbeddingGenerator:
    def __init__(self):
        self.batch_size = 32  # Default

        # For low VRAM GPUs:
        # self.batch_size = 8

        # For high-end GPUs:
        # self.batch_size = 64
```

### Query Result Limit

Default: `top_k=5`

```python
# More results (slower)
results = searcher.hybrid_search("query", top_k=20)

# Fewer results (faster)
results = searcher.hybrid_search("query", top_k=3)
```

### GPU vs CPU

**Force CPU** (useful for debugging):
```bash
export CUDA_VISIBLE_DEVICES=""
uv run process
```

**Force specific GPU**:
```bash
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
uv run process
```

**Check current device**:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
```

---

## Common Workflows 🔄

### Workflow 1: Add New Blog Post

```bash
# 1. Write blog post
vim blog/content/posts/2024/my-new-post.md

# 2. Add YAML frontmatter
---
title: "My New Post"
date: 2024-01-15
tags: ["python", "tutorial"]
---

# 3. Process all documents (includes new post)
uv run clean  # Optional: fresh start
uv run process

# 4. Test query
uv run query "new post topic"
```

### Workflow 2: Update Existing Document

```bash
# 1. Edit document
vim blog/content/posts/2024/existing-post.md

# 2. Rebuild database
uv run clean
uv run process

# 3. Verify changes
uv run query "updated content"
```

### Workflow 3: Experiment with Search Weights

```python
# experiment.py
from hackathon.database import get_db
from hackathon.retrieval import HybridSearcher

db = next(get_db())
searcher = HybridSearcher(db)

query = "database optimization"

# Test different weights
for bm25_w in [0.2, 0.4, 0.6, 0.8]:
    semantic_w = 1.0 - bm25_w

    results = searcher.hybrid_search(
        query,
        top_k=3,
        bm25_weight=bm25_w,
        semantic_weight=semantic_w
    )

    print(f"\n=== BM25: {bm25_w}, Semantic: {semantic_w} ===")
    for idx, r in enumerate(results, 1):
        print(f"{idx}. [{r.score:.4f}] {r.text_content[:60]}...")

db.close()
```

```bash
uv run python experiment.py
```

### Workflow 4: Export Search Results

```python
# export_results.py
import json
from hackathon.database import get_db
from hackathon.retrieval import HybridSearcher

db = next(get_db())
searcher = HybridSearcher(db)

# Queries to evaluate
queries = [
    "database pooling",
    "configure postgres",
    "python best practices"
]

# Run searches
all_results = {}
for query in queries:
    results = searcher.hybrid_search(query, top_k=5)

    all_results[query] = [
        {
            "rank": idx,
            "score": r.score,
            "text": r.text_content,
            "type": r.node_type,
            "metadata": r.metadata
        }
        for idx, r in enumerate(results, 1)
    ]

# Export to JSON
with open("search_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("✅ Results exported to search_results.json")

db.close()
```

```bash
uv run python export_results.py
```

---

## Debugging Tips 🐛

### Enable Verbose Logging

```python
# Add to your script
import logging

logging.basicConfig(level=logging.DEBUG)
```

### Inspect Search Scores

```python
results = searcher.hybrid_search("query", top_k=5)

for result in results:
    print(f"Score: {result.score}")
    print(f"Node ID: {result.node_id}")
    print(f"Type: {result.node_type}")
    print(f"Metadata: {result.metadata}")
    print(f"Text: {result.text_content[:100]}...")
    print("-" * 80)
```

### Check BM25 Tokens

```python
from hackathon.processing.bm25 import tokenize_text

query = "postgres pool_size configuration"
tokens = tokenize_text(query)
print(f"Tokens: {tokens}")
# Output: ['postgres', 'pool_size', 'configuration']
```

### Check Embedding Similarity

```python
from hackathon.processing.embedder import EmbeddingGenerator

embedder = EmbeddingGenerator()

query_vec = embedder.embed_text("database pooling")
chunk_vec = embedder.embed_text("connection pool configuration")

# Calculate cosine similarity
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarity = cosine_similarity(query_vec, chunk_vec)
print(f"Similarity: {similarity:.4f}")
# Output: Similarity: 0.8523 (high similarity!)
```

---

## Environment Variables 🔧

### Available Settings

| Variable | Purpose | Default |
|----------|---------|---------|
| `DATABASE_URL` | PostgreSQL connection | `postgresql://localhost/rag_system` |
| `ANTHROPIC_API_KEY` | API key for LLM | Required |
| `EMBEDDING_MODEL` | HuggingFace model | `ibm-granite/granite-embedding-30m-english` |
| `MARKDOWN_DIR` | Source directory | `./blog/content/posts` |
| `MARKDOWN_PATTERN` | File glob pattern | `**/*.md` |
| `MAX_CHUNK_SIZE` | Max chunk tokens | `512` |

### Overriding at Runtime

```bash
# Use different directory
MARKDOWN_DIR=./docs/guides uv run process

# Use different database
DATABASE_URL=postgresql://user:pass@remote:5432/rag uv run query
```

---

## API Reference 📚

### HybridSearcher

```python
from hackathon.retrieval import HybridSearcher

searcher = HybridSearcher(db)

# Perform search
results = searcher.hybrid_search(
    query="your query",
    top_k=10,
    bm25_weight=0.4,
    semantic_weight=0.6
)
# Returns: list[SearchResult]
```

### ContextExpander

```python
from hackathon.retrieval import ContextExpander

expander = ContextExpander(db)

# Get semantic block
block = expander.get_semantic_block_text(node)
# Returns: str | None

# Build hierarchical context
context = expander.build_context_text(
    node,
    depth=2,
    include_siblings=False,
    exclude_current=False
)
# Returns: str
```

### Database Operations

```python
from hackathon.database import (
    get_db,
    create_document,
    get_document_by_filename,
    get_all_leaf_nodes
)

# Get session
db = next(get_db())

# Create document
from hackathon.models.schemas import DocumentCreate

doc = create_document(db, DocumentCreate(
    filename="test.md",
    filepath="/path/to/test.md",
    metadata={"title": "Test"}
))

# Get all chunks
nodes = get_all_leaf_nodes(db)

db.close()
```

---

## Next Steps 📖

- [Testing Guide](testing.md) - Write tests for custom features
- [Contributing Guide](contributing.md) - Submit improvements
- [Architecture Overview](../architecture/overview.md) - Understand system design
