# Data Flow Documentation

This document traces how data flows through the system from ingestion to retrieval, with detailed ASCII diagrams.

## Complete Ingestion Pipeline

### Step-by-Step Data Transformation

```
INPUT: blog/posts/2025/my-post/index.md
├── Line 1-10:  YAML frontmatter
├── Line 11:    # Heading
├── Line 12-50: Content with code blocks
└── Line 51+:   More content

         │
         │ python-frontmatter.load()
         ▼

EXTRACTED DATA:
├── metadata: {title: "...", date: "...", tags: [...]}
└── clean_content: "# Heading\nContent..." (NO frontmatter)

         │
         │ Write to temp file
         ▼

TEMP FILE: /tmp/tmpXXX.md
└── Clean markdown (no YAML)

         │
         │ Docling DocumentConverter
         ▼

DOCLING DOCUMENT:
└── Structured document with:
    ├── Headings hierarchy
    ├── Text blocks
    ├── Code blocks
    ├── Lists
    └── Tables

         │
         │ Docling HybridChunker(tokenizer=granite)
         ▼

CHUNKS (Example):
┌─────────────────────────────────────────────┐
│ Chunk 0 (node_path="chunk_0")               │
│ ├── text: "I watch plenty of YouTube..."   │
│ ├── meta.headings: []                       │
│ ├── meta.doc_items: [text, text, text]     │
│ └── node_type: "paragraph"                  │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│ Chunk 1 (node_path="chunk_1")               │
│ ├── text: "The author offers... ```...```" │
│ ├── meta.headings: ["Installation & setup"]│
│ ├── meta.doc_items: [text, code, text...]  │
│ └── node_type: "code"                       │
└─────────────────────────────────────────────┘

         │
         │ For each chunk: extract_chunk_metadata()
         ▼

METADATA EXTRACTION:
├── Determine node_type from first non-text label
├── Collect doc_item_refs: ["#/texts/0", "#/texts/1", ...]
├── Collect doc_item_parents: ["#/body", "#/groups/2", ...]
├── Join headings: "Installation & setup"
├── Join docling_types: "text, code, text, code"
└── Store origin: "filename='index.md' binary_hash=..."

         │
         │ create_document_node()
         ▼

DATABASE INSERT:
INSERT INTO document_nodes (
    document_id,
    parent_id,  -- NULL for flat structure
    node_type,  -- "code", "paragraph", "list_item", etc.
    text_content,
    is_leaf,    -- TRUE (all chunks are leaves)
    node_path,  -- "chunk_0", "chunk_1", ...
    metadata    -- JSON with all extracted metadata
) VALUES (...);

         │
         │ Parallel indexing (3 streams)
         ▼

┌─────────────┐  ┌──────────────┐  ┌─────────────────┐
│   BM25      │  │  Embeddings  │  │  Contextual     │
│   Index     │  │  (GPU)       │  │  (LLM)          │
└─────────────┘  └──────────────┘  └─────────────────┘
```

## Detailed Pipeline Steps

### 1. YAML Frontmatter Extraction

**File**: `src/hackathon/processing/docling_processor.py:17`

```python
def extract_yaml_frontmatter(file_path: Path) -> tuple[str, dict]:
    post = frontmatter.load(file_path)
    metadata = {str(k): str(v) for k, v in post.metadata.items()}
    return post.content, metadata
```

**Input Example**:
```markdown
---
title: My Post
date: 2025-01-01
tags: [python, rag]
---

# Introduction
This is my post...
```

**Output**:
```python
(
    "# Introduction\nThis is my post...",  # clean_content
    {"title": "My Post", "date": "2025-01-01", "tags": "[python, rag]"}
)
```

### 2. Docling Document Conversion

**File**: `src/hackathon/processing/docling_processor.py:61`

```
┌──────────────────────────────────────┐
│ Input: Clean markdown (temp file)    │
└──────────────┬───────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │ DocumentConverter    │
    │ - Parse structure    │
    │ - Identify elements  │
    │ - Build doc tree     │
    └─────────┬────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ DoclingDocument                         │
│ ├── body                                │
│ │   ├── texts[0]: "# Introduction"     │
│ │   ├── texts[1]: "This is..."         │
│ │   ├── texts[2]: "```python..."       │
│ │   └── ...                             │
│ └── metadata                            │
│     ├── origin                          │
│     └── doc_hash                        │
└─────────────────────────────────────────┘
```

### 3. Hybrid Chunking

**File**: `src/hackathon/processing/docling_processor.py:78`

```
┌────────────────────────────────────────────┐
│ HybridChunker Configuration                │
│ ├── tokenizer: granite-embedding-30m       │
│ ├── max_tokens: ~512 (model dependent)     │
│ ├── strategy: Respect semantic boundaries  │
│ └── metadata: Preserve doc_items          │
└──────────────┬─────────────────────────────┘
               │
               ▼
     ┌─────────────────────┐
     │ Chunking Algorithm  │
     │                     │
     │ FOR each doc_item:  │
     │   IF fits in chunk: │
     │     Add to current  │
     │   ELSE:             │
     │     Start new chunk │
     │                     │
     │ Preserve:           │
     │ - Heading context   │
     │ - Doc item refs     │
     │ - Parent refs       │
     └─────────┬───────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ Chunk Metadata (Pydantic-like object)    │
│                                          │
│ class ChunkMeta:                         │
│   headings: list[str]                    │
│   doc_items: list[DocItem]               │
│   origin: DocOrigin                      │
│                                          │
│ class DocItem:                           │
│   self_ref: str  # "#/texts/5"           │
│   parent: RefItem  # {cref: "#/body"}    │
│   label: str  # "code", "text", "list"   │
└──────────────────────────────────────────┘
```

### 4. Metadata Extraction

**File**: `src/hackathon/processing/docling_processor.py:111`

```
INPUT: Docling Chunk
├── text: "Here is code:\n```python\nprint('hi')\n```"
├── meta.headings: ["Examples", "Python Code"]
└── meta.doc_items: [
      DocItem(self_ref="#/texts/10", parent="#/body", label="text"),
      DocItem(self_ref="#/texts/11", parent="#/groups/5", label="code")
    ]

         │
         │ _process_doc_items()
         ▼

EXTRACTED:
├── node_type: "code"  # First non-text label
├── docling_types: ["text", "code"]
├── doc_item_refs: ["#/texts/10", "#/texts/11"]
├── doc_item_parents: ["#/body", "#/groups/5"]
└── unique_parents: ["#/body", "#/groups/5"]

         │
         │ _build_metadata_dict()
         ▼

OUTPUT (JSON in database):
{
  "title": "My Post",
  "date": "2025-01-01",
  "headings": "Examples, Python Code",
  "docling_types": "text, code",
  "doc_item_refs": "#/texts/10, #/texts/11",
  "doc_item_parents": "#/body, #/groups/5",
  "origin": "mimetype='text/markdown' binary_hash=... filename='index.md'"
}
```

### 5. Database Storage

**File**: `src/hackathon/database/operations.py:77`

```
┌───────────────────────────────────────┐
│ DocumentNodeCreate (Pydantic)         │
├───────────────────────────────────────┤
│ document_id: int                      │
│ parent_id: None                       │
│ node_type: "code"                     │
│ text_content: "Here is code:..."      │
│ is_leaf: True                         │
│ node_path: "chunk_12"                 │
│ metadata: {...}  # JSON dict          │
└──────────────┬────────────────────────┘
               │
               ▼
     ┌─────────────────────┐
     │ SQLAlchemy ORM      │
     │                     │
     │ node = DocumentNode │
     │ db.add(node)        │
     │ db.flush()          │
     │ return node.id      │
     └─────────┬───────────┘
               │
               ▼
┌────────────────────────────────────────┐
│ PostgreSQL: document_nodes table       │
├────────────────────────────────────────┤
│ id: 42                                 │
│ document_id: 5                         │
│ parent_id: NULL                        │
│ node_type: 'code'                      │
│ text_content: 'Here is code:...'       │
│ is_leaf: true                          │
│ node_path: 'chunk_12'                  │
│ metadata: {"title": "...", ...}        │
│ created_at: 2025-01-15 10:30:00       │
└────────────────────────────────────────┘
```

## Indexing Data Flow

### BM25 Tokenization

**File**: `src/hackathon/processing/bm25.py:11`

```
INPUT: text_content
"Here is Python code: print('hello world')"

         │
         │ tokenize_text()
         ▼

TOKENIZATION:
├── Convert to lowercase
├── Remove punctuation
├── Split on whitespace
└── Filter tokens < 2 chars

OUTPUT: ['here', 'is', 'python', 'code', 'print', 'hello', 'world']

         │
         │ create_bm25_index_for_node()
         ▼

FOR EACH TOKEN:
INSERT INTO bm25_index (node_id, term, frequency)
VALUES
  (42, 'here', 1),
  (42, 'is', 1),
  (42, 'python', 1),
  (42, 'code', 1),
  (42, 'print', 1),
  (42, 'hello', 1),
  (42, 'world', 1);
```

### Embedding Generation

**File**: `src/hackathon/processing/embedder.py:45`

```
INPUT: Batch of (node_id, text) pairs
[
  (42, "Here is Python code: print('hello')"),
  (43, "This explains the syntax"),
  (44, "You can also use variables"),
]

         │
         │ model.encode(texts, batch_size=32)
         ▼

GPU PROCESSING:
┌─────────────────────────────────────┐
│ sentence-transformers/granite       │
│ ├── Tokenize text                   │
│ ├── Pass through transformer        │
│ ├── Pool token embeddings           │
│ └── Normalize to unit length        │
└──────────────┬──────────────────────┘
               │
               ▼

OUTPUT: numpy array (batch_size, 384)
[[0.123, -0.456, 0.789, ..., 0.234],   # 384 dims for node 42
 [0.234, -0.567, 0.890, ..., 0.345],   # 384 dims for node 43
 [0.345, -0.678, 0.901, ..., 0.456]]   # 384 dims for node 44

         │
         │ Convert to Python list, store in DB
         ▼

INSERT INTO embeddings (node_id, embedding)
VALUES
  (42, '[0.123, -0.456, ..., 0.234]'),
  (43, '[0.234, -0.567, ..., 0.345]'),
  (44, '[0.345, -0.678, ..., 0.456]');

Note: pgvector stores as VECTOR(384) type
```

### Contextual Summary Generation

**File**: `src/hackathon/processing/contextual.py:48`

```
INPUT: DocumentNode
├── text: "Option 2: Anthropic (Claude Opus)"
├── document_id: 5
└── metadata.headings: "Installation & setup"

         │
         │ build_context_from_ancestors()
         ▼

ANCESTOR CONTEXT:
"Document discusses Fabric setup.
Section 'Installation & setup' explains configuration menu."

         │
         │ Call Claude API with JSON mode
         ▼

LLM PROMPT:
```
You are summarizing a chunk within this context:

Document discusses Fabric setup.
Section 'Installation & setup' explains configuration menu.

Chunk content:
Option 2: Anthropic (Claude Opus)

Provide a concise 1-2 sentence summary that gives context.
Return JSON: {"context": "..."}
```

         │
         │ Claude responds
         ▼

LLM RESPONSE:
{
  "context": "In the Fabric setup configuration menu,
   option 2 allows selecting Anthropic's Claude Opus
   as the AI vendor for processing"
}

         │
         │ Strip common prefixes, store
         ▼

INSERT INTO contextual_chunks (node_id, contextual_summary)
VALUES (42, 'In the Fabric setup configuration menu...');
```

## Query Data Flow

### Hybrid Search Execution

**File**: `src/hackathon/retrieval/search.py:88`

```
INPUT: query = "How do I configure Fabric with Claude?"

         │
         ├──────────┬──────────┐
         │          │          │
         ▼          ▼          ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │ BM25   │ │Semantic│ │(Future)│
    │ Search │ │ Search │ │Rerank  │
    └────┬───┘ └───┬────┘ └────────┘
         │         │
         │         │
         ▼         ▼

BM25 SEARCH:
1. Tokenize query: ['configure', 'fabric', 'claude']
2. Load all BM25 terms into memory
3. Build BM25Okapi index
4. Score all documents:
   score = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * doc_len / avg_len))
5. Return top 100 with scores

SEMANTIC SEARCH:
1. Encode query to vector: model.encode(query)
   → [0.234, -0.456, 0.789, ..., 0.123]  # 384 dims

2. Query pgvector:
   SELECT node_id, 1 - (embedding <=> query_embedding) AS score
   FROM embeddings
   ORDER BY embedding <=> query_embedding
   LIMIT 100;

3. Return results with cosine similarity scores

         │
         ├─────────┴─────────┐
         │                   │
         ▼                   ▼
    BM25 Results        Semantic Results
    {node_42: 0.85}     {node_42: 0.92}
    {node_15: 0.72}     {node_38: 0.88}
    {node_28: 0.68}     {node_15: 0.75}
    ...                 ...

         │
         │ Merge with weights
         ▼

WEIGHTED SCORING:
for node_id in all_unique_nodes:
    bm25_score = bm25_results.get(node_id, 0) * 0.4
    semantic_score = semantic_results.get(node_id, 0) * 0.6
    final_score = bm25_score + semantic_score

RESULTS:
{node_42: 0.892}  # (0.85*0.4 + 0.92*0.6)
{node_38: 0.528}  # (0*0.4 + 0.88*0.6)
{node_15: 0.738}  # (0.72*0.4 + 0.75*0.6)
...

         │
         │ Sort by final_score, take top_k=5
         ▼

TOP 5 RESULTS:
1. node_42: 0.892
2. node_15: 0.738
3. node_38: 0.528
4. node_28: 0.472
5. node_51: 0.445
```

### Context Expansion

**File**: `src/hackathon/retrieval/context_expansion.py:163`

```
INPUT: Top result (node_42)

         │
         │ get_semantic_block_text()
         ▼

CHECK FOR SEMANTIC BLOCK:
1. Is this a code block? (node_type == "code")
2. Find related chunks:
   - Same heading: "Installation & setup"
   - Same parent_id: NULL
   - Also code type or 'code' in docling_types

FOUND: [node_41, node_42]  # Adjacent code chunks
         │
         │ Reassemble
         ▼
SEMANTIC BLOCK:
"```
> fabric --setup
[full setup menu with 27 options]
```"

         │
         │ build_context_text(depth=2, exclude_current=True)
         ▼

HIERARCHICAL CONTEXT:
(Empty in this case - no parent nodes, excluded current)

         │
         │ Display to user
         ▼

OUTPUT:
╭─── Full Context (with Semantic Block) ───╮
│ [Semantic Block]                          │
│ ```                                       │
│ > fabric --setup                          │
│ [full 27-option menu]                     │
│ ```                                       │
│                                           │
│ [Hierarchical Context]                    │
│ (empty - no parent structure)             │
╰───────────────────────────────────────────╯
```

## Error Handling Flow

```
┌─────────────────┐
│ Error Occurs    │
└────────┬────────┘
         │
    ┌────┴────────────────┬─────────────────┐
    │                     │                 │
    ▼                     ▼                 ▼
Docling Error      GPU Unavailable    LLM Timeout
    │                     │                 │
    │                     │                 │
    ▼                     ▼                 ▼
Log & Skip         CPU Fallback       Empty Context
Document           (with warning)     Continue Processing
    │                     │                 │
    │                     │                 │
    └──────────┬──────────┴─────────────────┘
               │
               ▼
       Continue Pipeline
       (System Resilient)
```

## Performance Characteristics

### Ingestion Throughput

```
Single Document Timeline:
├── YAML extraction:        ~1ms
├── Docling parsing:        ~100-500ms
├── Chunking:               ~50-200ms
├── Metadata extraction:    ~10ms
├── DB insert (chunks):     ~50ms
├── BM25 indexing:          ~20ms per chunk
├── Embedding (GPU):        ~100ms per batch (32 chunks)
└── Contextual (LLM):       ~500-2000ms per chunk

Total: ~1-5 seconds per document (depends on chunk count & LLM)
Bottleneck: LLM API calls for contextual summaries
```

### Query Latency

```
Single Query Timeline:
├── Query tokenization:     ~1ms
├── BM25 search:            ~50-200ms (in-memory)
├── Embedding generation:   ~20ms (GPU)
├── Vector search:          ~50-100ms (pgvector HNSW)
├── Score merging:          ~5ms
├── Context expansion:      ~20ms
└── Result formatting:      ~5ms

Total: ~150-350ms
Bottleneck: BM25 in-memory index rebuild for large corpora
```

## Data Volume Examples

### Small Corpus (100 docs)
```
Documents:        100
Avg chunks/doc:   ~20
Total chunks:     ~2,000

Database size:
├── document_nodes:      ~2MB
├── embeddings:          ~3MB (2000 * 384 * 4 bytes)
├── bm25_index:          ~1MB
└── contextual_chunks:   ~1MB

Total:                   ~7MB
Query time:              ~100ms
```

### Medium Corpus (5,000 docs)
```
Documents:        5,000
Avg chunks/doc:   ~20
Total chunks:     ~100,000

Database size:
├── document_nodes:      ~100MB
├── embeddings:          ~150MB
├── bm25_index:          ~50MB
└── contextual_chunks:   ~50MB

Total:                   ~350MB
Query time:              ~300ms (BM25 rebuild bottleneck)
```

### Large Corpus (50,000 docs)
```
Documents:        50,000
Avg chunks/doc:   ~20
Total chunks:     ~1,000,000

Database size:
├── document_nodes:      ~1GB
├── embeddings:          ~1.5GB
├── bm25_index:          ~500MB
└── contextual_chunks:   ~500MB

Total:                   ~3.5GB
Query time:              ~1-2s (needs optimization)

Recommendations:
- Switch from rank-bm25 to PostgreSQL FTS
- Add result caching
- Shard by document collection
```
