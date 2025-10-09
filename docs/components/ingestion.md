# Document Ingestion Pipeline

This document explains the complete document processing and ingestion pipeline, from raw markdown files to indexed, searchable chunks in the database.

## Overview

The ingestion pipeline transforms markdown documents into a searchable knowledge base by:

1. Extracting metadata (YAML frontmatter)
2. Parsing document structure (Docling)
3. Chunking content intelligently
4. Building multiple indexes (BM25, vector embeddings, contextual summaries)

## Pipeline Architecture

```
┌─────────────────┐
│ blog/*.md files │
└────────┬────────┘
         │
         ├─────────────────────────────────────┐
         │                                     │
         ▼                                     ▼
┌─────────────────────┐           ┌──────────────────────┐
│ YAML Frontmatter    │           │ Markdown Content     │
│ (python-frontmatter)│           │ (cleaned)            │
└────────┬────────────┘           └──────────┬───────────┘
         │                                   │
         │         ┌─────────────────────────┘
         │         │
         ▼         ▼
    ┌────────────────────────┐
    │ Docling Processing     │
    │ - Parse structure      │
    │ - Detect elements      │
    │ - Build hierarchy      │
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────────────┐
    │ HybridChunker          │
    │ - Respect boundaries   │
    │ - Max 512 tokens       │
    │ - Preserve context     │
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────────────┐
    │ Create DocumentNodes   │
    │ - Store in PostgreSQL  │
    │ - Build hierarchy      │
    │ - Extract metadata     │
    └────────┬───────────────┘
             │
             ├──────────┬──────────┬──────────────┐
             │          │          │              │
             ▼          ▼          ▼              ▼
    ┌────────────┐ ┌─────────┐ ┌──────────┐ ┌────────────┐
    │ BM25 Index │ │ Vector  │ │Contextual│ │ Commit DB  │
    │ (keyword)  │ │Embedding│ │ Summary  │ │            │
    └────────────┘ └─────────┘ └──────────┘ └────────────┘
```

## Step-by-Step Process

### Step 1: YAML Frontmatter Extraction

**File**: `src/hackathon/processing/docling_processor.py:extract_yaml_frontmatter()`

**Purpose**: Separate document metadata from content before processing.

**Process**:
```python
import frontmatter

def extract_yaml_frontmatter(file_path: Path) -> tuple[str, dict[str, str]]:
    post = frontmatter.load(file_path)
    metadata = {str(k): str(v) for k, v in post.metadata.items()}
    return post.content, metadata
```

**Input**:
```markdown
---
title: "Database Connection Pooling"
date: 2024-01-15
tags: ["postgresql", "performance"]
---

# Database Connection Pooling

Connection pooling is essential for...
```

**Output**:
- `clean_content`: Markdown without frontmatter
- `metadata`: `{"title": "Database Connection Pooling", "date": "2024-01-15", ...}`

**Why Important**: Docling processes pure markdown. YAML frontmatter would confuse its structure detection.

---

### Step 2: Docling Document Conversion

**File**: `src/hackathon/processing/docling_processor.py:_convert_document_with_docling()`

**Purpose**: Parse markdown structure and detect document elements.

**Process**:
```python
from docling.document_converter import DocumentConverter

def _convert_document_with_docling(file_path: Path):
    clean_content, _ = extract_yaml_frontmatter(file_path)

    # Write to temp file (Docling needs file path)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
        tmp.write(clean_content)
        tmp_path = Path(tmp.name)

    try:
        converter = DocumentConverter()
        result = converter.convert(str(tmp_path))
        return result.document
    finally:
        tmp_path.unlink(missing_ok=True)
```

**What Docling Detects**:
- Headings (hierarchy levels)
- Paragraphs
- Code blocks (with language)
- Lists (ordered/unordered, nested)
- Tables
- Block quotes

**Example Detection**:
```markdown
## Configuration

To configure pooling:

1. Set `pool_size` to 10
2. Enable `pool_pre_ping`

```python
pool = PoolManager(size=10)
```
```

Docling creates:
- Section header: "## Configuration"
- Paragraph: "To configure pooling:"
- List items: "1. Set pool_size..." (2 items)
- Code block: "python" (language detected)

---

### Step 3: Intelligent Chunking

**File**: `src/hackathon/processing/docling_processor.py:_chunk_document()`

**Purpose**: Split document into semantic chunks that fit within token limits.

**Process**:
```python
from docling_core.transforms.chunker import HybridChunker
from sentence_transformers import SentenceTransformer

def _chunk_document(doc):
    settings = get_settings()
    model = SentenceTransformer(settings.embedding_model, device="cpu")
    chunker = HybridChunker(tokenizer=model.tokenizer)
    return list(chunker.chunk(doc))
```

**Chunking Strategy**:

HybridChunker uses:
1. **Token counting**: Uses granite embedding model tokenizer
2. **Max tokens**: 512 (fits in embedding model)
3. **Boundary respect**: Never splits mid-sentence, mid-code-block, or mid-list
4. **Hierarchy preservation**: Includes heading context in metadata

**Example**:

**Original document**:
```markdown
## Database Pooling

Connection pooling reduces overhead. Benefits include:

- Faster connection reuse
- Reduced latency
- Better resource management

Here's how to configure it:

```python
from psycopg2.pool import ThreadedConnectionPool

pool = ThreadedConnectionPool(
    minconn=1,
    maxconn=10,
    dsn="postgresql://localhost/mydb"
)
```

The pool_size parameter controls...
```

**Chunks created**:

**Chunk 1** (~200 tokens):
```
Database Pooling

Connection pooling reduces overhead. Benefits include:

- Faster connection reuse
- Reduced latency
- Better resource management

Here's how to configure it:
```

**Chunk 2** (~180 tokens):
```python
from psycopg2.pool import ThreadedConnectionPool

pool = ThreadedConnectionPool(
    minconn=1,
    maxconn=10,
    dsn="postgresql://localhost/mydb"
)
```

**Chunk 3** (~50 tokens):
```
The pool_size parameter controls...
```

**Metadata per chunk**:
- `headings`: "Database Pooling" (inherited from section)
- `doc_items`: Types of content in chunk (e.g., "paragraph", "code", "list_item")
- `origin`: Source document elements

---

### Step 4: Node Creation in Database

**File**: `src/hackathon/processing/docling_processor.py:_create_nodes_from_chunks()`

**Purpose**: Store chunks as DocumentNode records with rich metadata.

**Process**:
```python
def _create_nodes_from_chunks(
    db: Session, document_id: int, chunks: list, frontmatter: dict[str, str]
) -> list[int]:
    leaf_node_ids = []

    for idx, chunk in enumerate(chunks):
        node_type, metadata = _extract_chunk_metadata(chunk, frontmatter)

        node_data = DocumentNodeCreate(
            document_id=document_id,
            text_content=chunk.text,
            node_type=node_type,
            node_path=f"chunk_{idx}",
            is_leaf=True,
            metadata=metadata,
        )

        node = create_document_node(db, node_data)
        leaf_node_ids.append(node.id)

    return leaf_node_ids
```

**Metadata Extraction**:

From Docling's chunk metadata, we extract:

```python
def _extract_chunk_metadata(chunk, frontmatter: dict[str, str]) -> tuple[str, dict]:
    meta = chunk.meta
    doc_items = getattr(meta, "doc_items", None) or []

    # Determine node type from doc_items
    node_type = "paragraph"
    docling_types = []

    for item in doc_items:
        if label := getattr(item, "label", None):
            label_str = str(label)
            docling_types.append(label_str)
            if node_type == "paragraph" and label_str != "text":
                node_type = label_str  # e.g., "code", "list_item", "table"

    # Build metadata dict
    chunk_metadata = {**frontmatter}  # Include YAML frontmatter

    if headings := getattr(meta, "headings", None):
        chunk_metadata["headings"] = ", ".join(str(h) for h in headings)

    if docling_types:
        chunk_metadata["docling_types"] = ", ".join(docling_types)

    return node_type, chunk_metadata
```

**Database Record Example**:

```sql
INSERT INTO document_nodes (
    document_id,
    text_content,
    node_type,
    node_path,
    is_leaf,
    metadata
) VALUES (
    42,
    'Connection pooling reduces overhead. Benefits include:...',
    'paragraph',
    'chunk_0',
    TRUE,
    '{"headings": "Database Pooling", "docling_types": "text, paragraph", "title": "Database Guide"}'::jsonb
);
```

---

### Step 5: BM25 Index Creation

**File**: `src/hackathon/processing/bm25.py:create_bm25_index_for_node()`

**Purpose**: Tokenize text for keyword search.

**Process**:
```python
def create_bm25_index_for_node(db: Session, node_id: int, text: str) -> None:
    tokens = tokenize_text(text)

    for token in tokens:
        bm25_entry = BM25IndexCreate(node_id=node_id, term=token)
        create_bm25_index(db, bm25_entry)

def tokenize_text(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return [t for t in tokens if len(t) > 1]  # Filter single chars
```

**Example**:

**Input**: `"Configure pool_size to 10 for optimal performance"`

**Tokens**: `["configure", "pool_size", "to", "10", "for", "optimal", "performance"]`

**Database records**:
```sql
INSERT INTO bm25_index (node_id, term) VALUES
(123, 'configure'),
(123, 'pool_size'),
(123, 'optimal'),
(123, 'performance');
-- "to", "for" removed as stopwords
```

**Why useful**: Fast exact keyword matching for technical terms like `pool_size`.

---

### Step 6: Vector Embedding Generation

**File**: `src/hackathon/processing/embedder.py:batch_create_embeddings()`

**Purpose**: Generate semantic vector representations for similarity search.

**Process**:
```python
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self):
        settings = get_settings()
        self.model = SentenceTransformer(
            settings.embedding_model,  # ibm-granite/granite-embedding-30m-english
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = 32

    def embed_text(self, text: str) -> list[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def batch_create_embeddings(self, db: Session, batch: list[tuple[int, str]]):
        node_ids = [node_id for node_id, _ in batch]
        texts = [text for _, text in batch]

        # Batch encode for efficiency
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        for node_id, embedding in zip(node_ids, embeddings):
            embedding_data = EmbeddingCreate(
                node_id=node_id,
                vector=embedding.tolist()
            )
            create_embedding(db, embedding_data)
```

**Embedding Example**:

**Input**: `"Configure pool_size to 10 for optimal performance"`

**Output**: 384-dimensional vector
```python
[0.023, -0.145, 0.891, ..., 0.234]  # 384 floats
```

**Storage**:
```sql
INSERT INTO embeddings (node_id, vector)
VALUES (123, '[0.023, -0.145, ...]'::vector(384));
```

**Why useful**: Captures semantic meaning. "database connection pooling" and "pool management" have similar vectors even without shared keywords.

---

### Step 7: Contextual Summary Generation

**File**: `src/hackathon/processing/contextual.py:create_contextual_chunk_for_node()`

**Purpose**: Use LLM to add context to chunks that lack it.

**Process**:
```python
class ContextualRetriever:
    def create_contextual_chunk_for_node(
        self, db: Session, node: DocumentNode, document_context: str
    ) -> ContextualChunk | None:
        # Build context from ancestors
        ancestors = get_node_ancestors(db, node.id)
        ancestor_text = "\n".join(a.text_content for a in ancestors[:3] if a.text_content)

        # Extract heading from metadata
        metadata = node.meta or {}
        heading = metadata.get("headings", "")

        context_parts = [document_context, heading, ancestor_text]
        context = " ".join(p for p in context_parts if p).strip()

        if not context:
            return None

        # Generate contextual summary with LLM
        summary = self._generate_context_with_llm(context, node.text_content)

        # Store in database
        chunk_data = ContextualChunkCreate(
            node_id=node.id,
            contextual_summary=summary
        )
        return create_contextual_chunk(db, chunk_data)
```

**LLM Prompt**:
```json
{
    "system": "You are a summarization assistant. Given context and a chunk, create a brief 1-2 sentence summary that situates the chunk within its context.",
    "context": "Database Guide > Connection Pooling",
    "chunk": "Set pool_size to 10 for optimal performance"
}
```

**LLM Response**:
```
"In the database connection pooling configuration, set the pool_size parameter to 10 for optimal performance."
```

**Database Record**:
```sql
INSERT INTO contextual_chunks (node_id, contextual_summary)
VALUES (123, 'In the database connection pooling configuration, set the pool_size parameter to 10 for optimal performance.');
```

**Why useful**: Improves retrieval accuracy. Query "how to configure postgres pool" now matches this chunk because the summary contains "database", "connection", "pooling", "configure", "pool" - terms missing from the original chunk.

---

## CLI Usage

### Processing All Documents

```bash
uv run process
```

**Output**:
```
Starting document processing...

Step 1: Processing markdown files
Processing documents: 100%|████████████| 25/25 [00:12<00:00,  2.08it/s]

Step 2: Building BM25 index
Indexing chunks: 100%|█████████████████| 342/342 [00:03<00:00, 114.67it/s]

Step 3: Generating embeddings
Embedding batches: 100%|██████████████| 11/11 [00:08<00:00,  1.29it/s]

Step 4: Generating contextual summaries
Contextual processing (doc 1): 100%|██| 15/15 [00:23<00:00,  1.57s/it]
Contextual processing (doc 2): 100%|██| 12/12 [00:19<00:00,  1.61s/it]
...

Processing complete!
Processed 25 documents with 342 chunks
```

### Clean and Rebuild

```bash
uv run clean   # Drop all tables
uv run process # Rebuild from scratch
```

---

## Performance Characteristics

| Stage | Speed | Bottleneck |
|-------|-------|------------|
| YAML extraction | ~1000 docs/sec | I/O (reading files) |
| Docling parsing | ~5-10 docs/sec | CPU (structure detection) |
| Chunking | ~10-20 docs/sec | CPU (tokenization) |
| BM25 indexing | ~100 chunks/sec | Database inserts |
| Embedding generation | ~5-20 chunks/sec | GPU throughput |
| Contextual summaries | ~1-2 chunks/sec | LLM API rate limits |

**Total**: ~1-5 documents/second for complete pipeline

---

## Error Handling

### Docling Parsing Errors

**Issue**: Malformed markdown or unsupported syntax

**Handling**:
```python
try:
    converter = DocumentConverter()
    result = converter.convert(str(tmp_path))
    return result.document
except Exception as e:
    logger.error(f"Docling conversion failed for {file_path}: {e}")
    raise  # Propagate to skip document
```

**Result**: Document is skipped, error logged, processing continues with next document.

### GPU Unavailable

**Issue**: CUDA/ROCm not available

**Handling**:
```python
self.model = SentenceTransformer(
    settings.embedding_model,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

if not torch.cuda.is_available():
    logger.warning("GPU not available, using CPU (will be slow)")
```

**Result**: Falls back to CPU with warning. Embeddings will be ~10-50x slower.

### LLM API Timeout

**Issue**: Anthropic API timeout or rate limit

**Handling**:
```python
try:
    response = client.messages.create(...)
    return response.content[0].text
except anthropic.APIError as e:
    logger.error(f"LLM API error: {e}")
    return None  # Store empty contextual chunk
```

**Result**: Contextual summary skipped for that chunk. Search still works with BM25 + embeddings.

---

## Monitoring and Debugging

### Enable Verbose Logging

```python
# In src/hackathon/utils/logging.py
import logging

logger = logging.getLogger("hackathon")
logger.setLevel(logging.DEBUG)  # Change from INFO
```

### Check Database Records

```sql
-- Count chunks per document
SELECT d.filename, COUNT(dn.id) as chunk_count
FROM documents d
LEFT JOIN document_nodes dn ON d.id = dn.document_id
GROUP BY d.id, d.filename;

-- Check indexing coverage
SELECT
    COUNT(DISTINCT dn.id) as total_chunks,
    COUNT(DISTINCT e.node_id) as embedded_chunks,
    COUNT(DISTINCT b.node_id) as bm25_indexed_chunks,
    COUNT(DISTINCT c.node_id) as contextual_chunks
FROM document_nodes dn
LEFT JOIN embeddings e ON dn.id = e.node_id
LEFT JOIN bm25_index b ON dn.id = b.node_id
LEFT JOIN contextual_chunks c ON dn.id = c.node_id
WHERE dn.is_leaf = TRUE;
```

### Inspect Chunk Metadata

```sql
SELECT
    id,
    node_type,
    node_path,
    metadata->>'headings' as headings,
    metadata->>'docling_types' as docling_types,
    LEFT(text_content, 100) as preview
FROM document_nodes
WHERE document_id = 1
ORDER BY id;
```

---

## Next Steps

- [Search & Retrieval](search.md) - How queries use these indexes
- [Context Expansion](context-expansion.md) - How results are enriched
- [Database Schema](../architecture/database.md) - Table definitions
