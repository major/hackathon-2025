# Database Schema

This document details the PostgreSQL database schema, table relationships, and indexing strategy.

## Entity Relationship Diagram

```
┌─────────────────────────────────────────────┐
│              documents                       │
├─────────────────────────────────────────────┤
│ PK │ id                    SERIAL            │
│    │ filename              VARCHAR(255) UQ   │
│    │ filepath              VARCHAR(512)      │
│    │ processing_timestamp  TIMESTAMPTZ       │
│    │ metadata              JSON              │
└─────┬───────────────────────────────────────┘
      │
      │ 1:N
      │
┌─────▼───────────────────────────────────────┐
│           document_nodes                     │
├─────────────────────────────────────────────┤
│ PK │ id                    SERIAL            │
│ FK │ document_id           INTEGER           │
│ FK │ parent_id             INTEGER (NULL)    │
│    │ node_type             VARCHAR(50)       │
│    │ text_content          TEXT              │
│    │ is_leaf               BOOLEAN           │
│    │ node_path             VARCHAR(255) UQ   │
│    │ metadata              JSON              │
│    │ created_at            TIMESTAMPTZ       │
└─────┬─┬─────────────────────────────────────┘
      │ │
      │ └────────┐
      │          │ self-referential (parent_id → id)
  ┌───┴──┐   ┌───┴──┐   ┌────────┐
  │      │   │      │   │        │
  ▼      ▼   ▼      ▼   ▼        ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────────────┐
│embed-│ │ bm25 │ │contx │ │   (future)   │
│dings │ │_index│ │_chnks│ │   tables     │
└──────┘ └──────┘ └──────┘ └──────────────┘
```

## Table Definitions

### documents

Stores source document metadata.

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL UNIQUE,
    filepath VARCHAR(512) NOT NULL,
    processing_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX idx_documents_filename ON documents(filename);
CREATE INDEX idx_documents_timestamp ON documents(processing_timestamp DESC);
```

**Columns**:

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL | Primary key |
| `filename` | VARCHAR(255) | Unique document identifier (relative path) |
| `filepath` | VARCHAR(512) | Full filesystem path |
| `processing_timestamp` | TIMESTAMPTZ | When document was ingested |
| `metadata` | JSONB | YAML frontmatter + additional metadata |

**Sample Data**:
```sql
{
  "id": 5,
  "filename": "blog/posts/2025/my-post/index.md",
  "filepath": "/home/user/blog/posts/2025/my-post/index.md",
  "processing_timestamp": "2025-01-15T10:30:00Z",
  "metadata": {
    "title": "My Post",
    "date": "2025-01-15",
    "tags": ["python", "rag"],
    "author": "John Doe"
  }
}
```

### document_nodes

Stores hierarchical document chunks.

```sql
CREATE TABLE document_nodes (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_id INTEGER REFERENCES document_nodes(id) ON DELETE CASCADE,
    node_type VARCHAR(50) NOT NULL,
    text_content TEXT,
    is_leaf BOOLEAN NOT NULL DEFAULT FALSE,
    node_path VARCHAR(255) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_node_path UNIQUE (document_id, node_path)
);

CREATE INDEX idx_nodes_document ON document_nodes(document_id);
CREATE INDEX idx_nodes_parent ON document_nodes(parent_id);
CREATE INDEX idx_nodes_type ON document_nodes(node_type);
CREATE INDEX idx_nodes_leaf ON document_nodes(is_leaf) WHERE is_leaf = TRUE;
CREATE INDEX idx_nodes_metadata ON document_nodes USING GIN (metadata);
```

**Columns**:

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL | Primary key |
| `document_id` | INTEGER | Foreign key to documents |
| `parent_id` | INTEGER | Self-referential FK (NULL for root nodes) |
| `node_type` | VARCHAR(50) | Docling classification: paragraph, code, list_item, etc. |
| `text_content` | TEXT | Actual chunk text |
| `is_leaf` | BOOLEAN | TRUE if indexable chunk, FALSE if container |
| `node_path` | VARCHAR(255) | Unique path within document: "chunk_0", "chunk_1", etc. |
| `metadata` | JSONB | Docling metadata: headings, doc_items, origin, etc. |
| `created_at` | TIMESTAMPTZ | Chunk creation time |

**Sample Data**:
```sql
{
  "id": 42,
  "document_id": 5,
  "parent_id": NULL,
  "node_type": "code",
  "text_content": "Here is Python code:\n```python\nprint('hello')\n```",
  "is_leaf": TRUE,
  "node_path": "chunk_12",
  "metadata": {
    "headings": "Examples, Python Code",
    "docling_types": "text, code",
    "doc_item_refs": "#/texts/10, #/texts/11",
    "doc_item_parents": "#/body, #/groups/5",
    "origin": "mimetype='text/markdown' binary_hash=12345 filename='index.md'"
  },
  "created_at": "2025-01-15T10:30:15Z"
}
```

### embeddings

Stores vector embeddings for semantic search.

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    node_id INTEGER NOT NULL REFERENCES document_nodes(id) ON DELETE CASCADE,
    embedding VECTOR(384) NOT NULL,  -- granite-embedding-30m-english dimension
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_node_embedding UNIQUE (node_id)
);

-- HNSW index for approximate nearest neighbor search
CREATE INDEX idx_embeddings_vector ON embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_embeddings_node ON embeddings(node_id);
```

**Columns**:

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL | Primary key |
| `node_id` | INTEGER | Foreign key to document_nodes (1:1 relationship) |
| `embedding` | VECTOR(384) | 384-dimensional embedding vector |
| `created_at` | TIMESTAMPTZ | Embedding generation time |

**Index Strategy**:
- **HNSW (Hierarchical Navigable Small World)**: Approximate nearest neighbor search
  - `m = 16`: Max connections per layer (trade-off: recall vs. speed)
  - `ef_construction = 64`: Build-time search depth (trade-off: index quality vs. build time)
  - `vector_cosine_ops`: Use cosine distance (1 - cosine_similarity)

**Query Example**:
```sql
-- Find top 10 most similar chunks
SELECT
    node_id,
    1 - (embedding <=> '[0.123, -0.456, ...]'::vector) AS similarity
FROM embeddings
ORDER BY embedding <=> '[0.123, -0.456, ...]'::vector
LIMIT 10;
```

### bm25_index

Stores tokenized terms for BM25 search.

```sql
CREATE TABLE bm25_index (
    id SERIAL PRIMARY KEY,
    node_id INTEGER NOT NULL REFERENCES document_nodes(id) ON DELETE CASCADE,
    term VARCHAR(100) NOT NULL,
    frequency INTEGER NOT NULL DEFAULT 1,

    CONSTRAINT unique_node_term UNIQUE (node_id, term)
);

CREATE INDEX idx_bm25_term ON bm25_index(term);
CREATE INDEX idx_bm25_node ON bm25_index(node_id);
CREATE INDEX idx_bm25_term_freq ON bm25_index(term, frequency);
```

**Columns**:

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL | Primary key |
| `node_id` | INTEGER | Foreign key to document_nodes |
| `term` | VARCHAR(100) | Lowercase, punctuation-removed token |
| `frequency` | INTEGER | Term frequency in this node |

**Sample Data**:
```sql
INSERT INTO bm25_index (node_id, term, frequency) VALUES
(42, 'python', 2),
(42, 'code', 1),
(42, 'print', 1),
(42, 'hello', 1);
```

**Note**: Current implementation loads all terms into memory and uses rank-bm25 library. For >10K documents, consider PostgreSQL FTS instead.

### contextual_chunks

Stores LLM-generated contextual summaries.

```sql
CREATE TABLE contextual_chunks (
    id SERIAL PRIMARY KEY,
    node_id INTEGER NOT NULL REFERENCES document_nodes(id) ON DELETE CASCADE,
    contextual_summary TEXT NOT NULL,
    document_context TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_node_context UNIQUE (node_id)
);

CREATE INDEX idx_contextual_node ON contextual_chunks(node_id);
CREATE INDEX idx_contextual_summary_fts ON contextual_chunks
    USING GIN (to_tsvector('english', contextual_summary));
```

**Columns**:

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL | Primary key |
| `node_id` | INTEGER | Foreign key to document_nodes (1:1 relationship) |
| `contextual_summary` | TEXT | LLM-generated summary with context |
| `document_context` | TEXT | Original document/section context (rarely used) |
| `created_at` | TIMESTAMPTZ | Summary generation time |

**Sample Data**:
```sql
{
  "id": 18,
  "node_id": 42,
  "contextual_summary": "In a Python tutorial section on examples, this code snippet demonstrates printing a hello world message using the print function.",
  "document_context": "Document about Python basics",
  "created_at": "2025-01-15T10:32:00Z"
}
```

## Relationship Patterns

### Hierarchical Structure (Currently Unused)

The schema supports hierarchical document structure via `parent_id`:

```
document_nodes
├── id: 1, parent_id: NULL, type: "document", is_leaf: FALSE
│   ├── id: 2, parent_id: 1, type: "section", is_leaf: FALSE
│   │   ├── id: 3, parent_id: 2, type: "paragraph", is_leaf: TRUE
│   │   └── id: 4, parent_id: 2, type: "code", is_leaf: TRUE
│   └── id: 5, parent_id: 1, type: "section", is_leaf: FALSE
│       └── id: 6, parent_id: 5, type: "list_item", is_leaf: TRUE
```

**Current Implementation**: All chunks have `parent_id = NULL` (flat structure). Docling metadata provides structural information instead.

### One-to-One Relationships

```
document_nodes (1) ──< (1) embeddings
document_nodes (1) ──< (1) contextual_chunks
```

Each leaf node has exactly one embedding and one contextual summary.

### One-to-Many Relationships

```
documents (1) ──< (N) document_nodes
document_nodes (1) ──< (N) bm25_index (multiple terms per node)
```

## Query Patterns

### Full-Text Search (BM25)

```sql
-- Find all nodes containing term "python"
SELECT DISTINCT n.id, n.text_content
FROM document_nodes n
JOIN bm25_index b ON n.id = b.node_id
WHERE b.term = 'python';
```

**Current Limitation**: BM25 scoring happens in-memory (Python), not in PostgreSQL.

### Semantic Search

```sql
-- Find semantically similar chunks
WITH query_embedding AS (
    SELECT '[0.123, -0.456, ..., 0.789]'::vector AS vec
)
SELECT
    n.id,
    n.text_content,
    1 - (e.embedding <=> q.vec) AS similarity
FROM document_nodes n
JOIN embeddings e ON n.id = e.node_id
CROSS JOIN query_embedding q
ORDER BY e.embedding <=> q.vec
LIMIT 10;
```

### Ancestor Retrieval

```sql
-- Get all ancestors of a node (recursive CTE)
WITH RECURSIVE ancestors AS (
    SELECT id, parent_id, text_content, 0 AS depth
    FROM document_nodes
    WHERE id = 42

    UNION ALL

    SELECT n.id, n.parent_id, n.text_content, a.depth + 1
    FROM document_nodes n
    JOIN ancestors a ON n.id = a.parent_id
)
SELECT * FROM ancestors ORDER BY depth DESC;
```

### Children Retrieval

```sql
-- Get all direct children of a node
SELECT id, node_type, text_content
FROM document_nodes
WHERE parent_id = 42
ORDER BY node_path;
```

### Metadata Queries

```sql
-- Find all code blocks with specific heading
SELECT id, text_content
FROM document_nodes
WHERE node_type = 'code'
AND metadata->>'headings' LIKE '%Installation%';

-- Find chunks by doc_item_parent
SELECT id, text_content
FROM document_nodes
WHERE metadata->>'doc_item_parents' LIKE '%#/groups/5%';
```

## Index Strategy

### When to Create Indexes

✅ **Index**:
- Foreign keys: `document_id`, `parent_id`, `node_id`
- Frequently filtered columns: `node_type`, `is_leaf`
- Sort columns: `created_at`, `processing_timestamp`
- Search columns: `term` (bm25_index), `embedding` (HNSW)

❌ **Don't Index**:
- Large text columns: `text_content`, `contextual_summary` (unless using FTS)
- Low cardinality: `is_leaf` (only 2 values, but we use partial index)
- Rarely queried: `filepath`

### Index Maintenance

```sql
-- Check index usage
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;

-- Rebuild HNSW index if needed (after bulk inserts)
REINDEX INDEX idx_embeddings_vector;

-- Analyze tables for query planner
ANALYZE documents;
ANALYZE document_nodes;
ANALYZE embeddings;
ANALYZE bm25_index;
ANALYZE contextual_chunks;
```

## Performance Tuning

### Vacuum Strategy

```sql
-- Auto-vacuum is usually sufficient, but manual vacuum for large deletes
VACUUM ANALYZE document_nodes;

-- Full vacuum to reclaim disk space (locks table)
VACUUM FULL document_nodes;
```

### Connection Pooling

SQLAlchemy uses connection pooling by default:

```python
engine = create_engine(
    settings.database_url,
    pool_size=10,          # Maintain 10 idle connections
    max_overflow=20,       # Allow 20 additional connections when pool exhausted
    pool_pre_ping=True,    # Verify connections before use
)
```

### Query Optimization

**Problem**: N+1 query when fetching nodes + embeddings

```python
# ❌ Bad: N+1 queries
nodes = db.query(DocumentNode).all()
for node in nodes:
    embedding = db.query(Embedding).filter_by(node_id=node.id).first()
```

**Solution**: Eager loading with joins

```python
# ✅ Good: Single query
nodes = db.query(DocumentNode)\
    .outerjoin(Embedding)\
    .options(joinedload(DocumentNode.embeddings))\
    .all()
```

## Migration Strategy

### Adding New Tables

1. Create migration script:
```sql
-- migrations/002_add_reranking.sql
CREATE TABLE reranking_scores (
    id SERIAL PRIMARY KEY,
    node_id INTEGER NOT NULL REFERENCES document_nodes(id),
    query_hash VARCHAR(64) NOT NULL,
    score FLOAT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_reranking_node ON reranking_scores(node_id);
CREATE INDEX idx_reranking_query ON reranking_scores(query_hash);
```

2. Apply with SQLAlchemy:
```python
with engine.begin() as conn:
    conn.execute(text(open('migrations/002_add_reranking.sql').read()))
```

### Modifying Existing Tables

**Example**: Add `chunk_strategy` column

```sql
ALTER TABLE document_nodes
ADD COLUMN chunk_strategy VARCHAR(50) DEFAULT 'docling';

UPDATE document_nodes
SET chunk_strategy = 'docling'
WHERE chunk_strategy IS NULL;

ALTER TABLE document_nodes
ALTER COLUMN chunk_strategy SET NOT NULL;
```

## Backup & Recovery

### Backup

```bash
# Full database backup
pg_dump -Fc hackathon_db > backup_$(date +%Y%m%d).dump

# Schema only
pg_dump -s hackathon_db > schema.sql

# Data only (specific tables)
pg_dump -t document_nodes -t embeddings hackathon_db > data.sql
```

### Restore

```bash
# Restore full database
pg_restore -d hackathon_db backup_20250115.dump

# Restore schema + data
psql hackathon_db < schema.sql
psql hackathon_db < data.sql
```

## Monitoring Queries

### Database Size

```sql
SELECT
    pg_size_pretty(pg_database_size('hackathon_db')) AS total_size,
    pg_size_pretty(pg_total_relation_size('document_nodes')) AS nodes_size,
    pg_size_pretty(pg_total_relation_size('embeddings')) AS embeddings_size,
    pg_size_pretty(pg_total_relation_size('bm25_index')) AS bm25_size;
```

### Table Statistics

```sql
SELECT
    schemaname,
    tablename,
    n_tup_ins AS inserts,
    n_tup_upd AS updates,
    n_tup_del AS deletes,
    n_live_tup AS live_rows,
    n_dead_tup AS dead_rows
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;
```

### Slow Queries

```sql
-- Enable query logging in postgresql.conf
-- log_min_duration_statement = 100  # Log queries > 100ms

-- Check pg_stat_statements extension
SELECT
    query,
    calls,
    mean_exec_time,
    max_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```
