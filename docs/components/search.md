# Search & Retrieval System

This document explains how the hybrid search system finds and ranks relevant document chunks using a combination of keyword matching (BM25) and semantic similarity (vector search).

## Overview 🎯

The search system combines two complementary strategies:

- **BM25** 📝: Traditional keyword matching (finds exact technical terms)
- **Semantic Search** 🧠: Meaning-based similarity (finds conceptually related content)
- **Hybrid Ranking** ⚖️: Weighted combination (40% BM25 + 60% semantic)

## Architecture 🏗️

```
┌──────────────────┐
│   User Query     │
│ "postgres pool"  │
└────────┬─────────┘
         │
         ├─────────────────────────────────┐
         │                                 │
         ▼                                 ▼
┌──────────────────┐           ┌───────────────────────┐
│  BM25 Search     │           │  Semantic Search      │
│  ┌────────────┐  │           │  ┌─────────────────┐  │
│  │ Tokenize   │  │           │  │ Encode query    │  │
│  │ query      │  │           │  │ to vector       │  │
│  └─────┬──────┘  │           │  └────────┬────────┘  │
│        │         │           │           │           │
│        ▼         │           │           ▼           │
│  ┌────────────┐  │           │  ┌─────────────────┐  │
│  │ Load BM25  │  │           │  │ pgvector ANN    │  │
│  │ index      │  │           │  │ search          │  │
│  └─────┬──────┘  │           │  └────────┬────────┘  │
│        │         │           │           │           │
│        ▼         │           │           ▼           │
│  ┌────────────┐  │           │  ┌─────────────────┐  │
│  │ Score docs │  │           │  │ Cosine distance │  │
│  │ (TF-IDF)   │  │           │  │ ranking         │  │
│  └─────┬──────┘  │           │  └────────┬────────┘  │
│        │         │           │           │           │
│        ▼         │           │           ▼           │
│  ┌────────────┐  │           │  ┌─────────────────┐  │
│  │ Top-20     │  │           │  │ Top-20          │  │
│  │ results    │  │           │  │ results         │  │
│  └────────────┘  │           │  └─────────────────┘  │
└─────────┬────────┘           └──────────┬────────────┘
          │                               │
          └────────────┬──────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  Merge & Normalize     │
          │  - Union all node IDs  │
          │  - Normalize scores    │
          │  - Weighted combine    │
          └────────────┬───────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  Sort by combined score│
          │  Return top-K          │
          └────────────┬───────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  Fetch node details    │
          │  - text_content        │
          │  - metadata            │
          │  - document info       │
          └────────────────────────┘
```

## BM25 Keyword Search 📝

### What is BM25?

**BM25** (Best Matching 25) is an algorithm that scores documents based on keyword frequency, with diminishing returns for repeated terms and length normalization.

**Formula** (simplified):
```
score(doc, query) = Σ IDF(term) × (TF(term) × (k1 + 1)) / (TF(term) + k1 × (1 - b + b × doc_length/avg_length))

Where:
- IDF = Inverse Document Frequency (rarity of term across corpus)
- TF = Term Frequency (how often term appears in document)
- k1 = 1.5 (term frequency saturation)
- b = 0.75 (length normalization factor)
```

### Implementation

**File**: `src/hackathon/retrieval/search.py:bm25_search()`

```python
from rank_bm25 import BM25Okapi

def bm25_search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
    # Load all BM25 index entries from database
    bm25_indices = self.db.execute(select(BM25Index)).scalars().all()

    # Group terms by node
    node_terms = {}
    for entry in bm25_indices:
        if entry.node_id not in node_terms:
            node_terms[entry.node_id] = []
        node_terms[entry.node_id].append(entry.term)

    # Build corpus (list of token lists)
    corpus = [node_terms.get(idx.node_id, []) for idx in bm25_indices]

    # Create BM25 index
    bm25 = BM25Okapi(corpus)

    # Tokenize query
    query_tokens = tokenize_text(query)

    # Get scores
    scores = bm25.get_scores(query_tokens)

    # Get top k
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [(bm25_indices[i].node_id, float(scores[i])) for i in top_indices if scores[i] > 0]
```

### BM25 Example Walkthrough

**Query**: `"postgres pool configuration"`

**Step 1: Tokenize**
```python
query_tokens = ["postgres", "pool", "configuration"]
```

**Step 2: Load corpus**
```python
# Document 1 (node 42)
["postgresql", "database", "connection", "pool", "config", "pool_size", "10"]

# Document 2 (node 43)
["postgres", "pool", "management", "setup", "configuration"]

# Document 3 (node 44)
["python", "flask", "application", "deployment"]
```

**Step 3: Calculate IDF** (inverse document frequency)
```python
IDF("postgres") = log(3 / 1) = 1.10  # appears in 1/3 docs (rare-ish)
IDF("pool") = log(3 / 2) = 0.40      # appears in 2/3 docs (common)
IDF("configuration") = log(3 / 1) = 1.10  # appears in 1/3 docs
```

**Step 4: Calculate TF-IDF scores**

**Document 1 (node 42)**:
```
TF("postgres") = 0 → score = 0
TF("pool") = 2 → score = 0.40 × (2 × 2.5) / (2 + 1.5 × 0.9) ≈ 1.2
TF("configuration") = 0 → score = 0
Total: 1.2
```

**Document 2 (node 43)**:
```
TF("postgres") = 1 → score = 1.10 × (1 × 2.5) / (1 + 1.5 × 0.9) ≈ 1.1
TF("pool") = 1 → score = 0.40 × (1 × 2.5) / (1 + 1.5 × 0.9) ≈ 0.4
TF("configuration") = 1 → score = 1.10 × (1 × 2.5) / (1 + 1.5 × 0.9) ≈ 1.1
Total: 2.6 ⭐ Highest!
```

**Document 3 (node 44)**:
```
TF("postgres") = 0 → score = 0
TF("pool") = 0 → score = 0
TF("configuration") = 0 → score = 0
Total: 0
```

**Step 5: Rank results**
```python
[
    (43, 2.6),  # Document 2 - best match
    (42, 1.2),  # Document 1 - partial match
]
```

### BM25 Strengths & Weaknesses

**✅ Strengths**:
- Fast (in-memory scoring)
- Excellent for technical terms (`pool_size`, `maxconn`, `SIGHUP`)
- Exact keyword matching
- No GPU required

**❌ Weaknesses**:
- Misses synonyms ("connection pool" vs "database pooling")
- No semantic understanding
- Fails on paraphrases ("how to configure pool" vs "pool configuration guide")

---

## Semantic Vector Search 🧠

### What are Vector Embeddings?

Vector embeddings are high-dimensional numerical representations of text that capture semantic meaning.

**Analogy**: Like GPS coordinates for concepts
- Similar meanings → close coordinates
- Different meanings → far apart

**Example**:
```python
# 384-dimensional vectors (simplified to 3D for visualization)

"database pooling"     → [0.8,  0.3, -0.1]
"connection pool"      → [0.75, 0.35, -0.05]  # Close! Similar meaning
"cooking recipe"       → [-0.2, 0.9,  0.7]    # Far! Different meaning

cosine_similarity([0.8, 0.3, -0.1], [0.75, 0.35, -0.05]) = 0.95  # Very similar!
cosine_similarity([0.8, 0.3, -0.1], [-0.2, 0.9, 0.7])    = 0.12  # Unrelated
```

### Implementation

**File**: `src/hackathon/retrieval/search.py:semantic_search()`

```python
from sentence_transformers import SentenceTransformer

def semantic_search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
    # Generate query embedding
    query_vector = self.embedding_generator.embed_text(query)

    # Query embeddings with vector similarity (using pgvector)
    stmt = (
        select(
            Embedding.node_id,
            Embedding.vector.cosine_distance(query_vector).label("distance")
        )
        .order_by("distance")
        .limit(top_k)
    )

    results = self.db.execute(stmt).all()

    # Convert distance to similarity score (1 - distance for cosine)
    return [(node_id, 1 - float(distance)) for node_id, distance in results]
```

### Semantic Search Example

**Query**: `"how to optimize database connections"`

**Step 1: Encode query**
```python
query_vector = model.encode("how to optimize database connections")
# → [0.12, -0.34, 0.78, ..., 0.45]  (384 dimensions)
```

**Step 2: Find nearest neighbors in vector space**

**Database contains** (simplified):

```python
# Node 42: "Connection pooling reduces overhead by reusing connections"
embedding_42 = [0.15, -0.30, 0.82, ..., 0.50]
cosine_distance(query, embedding_42) = 0.08  # Close!

# Node 43: "Pool configuration with pool_size parameter"
embedding_43 = [0.18, -0.28, 0.75, ..., 0.48]
cosine_distance(query, embedding_43) = 0.12  # Moderately close

# Node 44: "Python Flask application deployment"
embedding_44 = [-0.50, 0.60, -0.20, ..., -0.10]
cosine_distance(query, embedding_44) = 0.85  # Far!
```

**Step 3: Convert distance to similarity**
```python
[
    (42, 1 - 0.08 = 0.92),  # High similarity!
    (43, 1 - 0.12 = 0.88),  # Good similarity
    (44, 1 - 0.85 = 0.15),  # Low similarity
]
```

**Step 4: Rank by similarity**
```python
[
    (42, 0.92),  # Best match
    (43, 0.88),  # Second best
]
```

### Semantic Search Strengths & Weaknesses

**✅ Strengths**:
- Understands meaning (finds "connection pooling" for "optimize database connections")
- Handles synonyms and paraphrases
- Language-agnostic (works across different phrasings)

**❌ Weaknesses**:
- May miss exact technical terms
- Requires GPU for fast encoding
- No explainability (why did it match?)

---

## Hybrid Search ⚖️

### Why Combine Both?

Different query types benefit from different strategies:

| Query Type | Best Strategy | Example |
|------------|---------------|---------|
| Exact technical term | BM25 | "configure pool_size parameter" |
| Conceptual question | Semantic | "how to manage database connections" |
| Mixed | **Hybrid** | "pool_size configuration guide" |

### Weighted Combination

**File**: `src/hackathon/retrieval/search.py:hybrid_search()`

**Default weights**:
- BM25: 40%
- Semantic: 60%

```python
def hybrid_search(
    self,
    query: str,
    top_k: int = 10,
    bm25_weight: float = 0.4,
    semantic_weight: float = 0.6,
) -> list[SearchResult]:
    # Get BM25 results
    bm25_results = dict(self.bm25_search(query, top_k=top_k * 2))

    # Get semantic results
    semantic_results = dict(self.semantic_search(query, top_k=top_k * 2))

    # Combine scores
    all_node_ids = set(bm25_results.keys()) | set(semantic_results.keys())
    combined_scores = {}

    # Normalize scores to 0-1 range
    bm25_scores = list(bm25_results.values())
    max_bm25 = max(bm25_scores) if bm25_scores else 1.0

    for node_id in all_node_ids:
        bm25_score = bm25_results.get(node_id, 0.0) / max_bm25 if max_bm25 > 0 else 0.0
        semantic_score = semantic_results.get(node_id, 0.0)

        combined_score = (bm25_weight * bm25_score) + (semantic_weight * semantic_score)
        combined_scores[node_id] = combined_score

    # Sort by combined score
    sorted_node_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Fetch node details
    results = []
    for node_id, score in sorted_node_ids:
        stmt = select(DocumentNode).where(DocumentNode.id == node_id)
        node = self.db.execute(stmt).scalar_one_or_none()

        if node:
            results.append(SearchResult(
                node_id=node.id,
                score=score,
                text_content=node.text_content or "",
                node_type=node.node_type,
                node_path=node.node_path,
                document_id=node.document_id,
                metadata=node.meta or {},
            ))

    return results
```

### Hybrid Search Example

**Query**: `"postgres pool_size configuration"`

**BM25 Results** (raw):
```python
{
    42: 3.2,  # Has "postgres", "pool", mentions "configuration"
    43: 5.8,  # Has "postgres", "pool_size", "configuration" ⭐ All keywords!
    44: 1.1,  # Only has "configuration"
}
```

**Semantic Results** (cosine similarity):
```python
{
    42: 0.92,  # ⭐ Excellent semantic match (detailed pooling explanation)
    43: 0.78,  # Good match (configuration focused)
    45: 0.85,  # Strong match (general database optimization, includes pooling)
}
```

**Step 1: Normalize BM25 scores**
```python
max_bm25 = 5.8

normalized_bm25 = {
    42: 3.2 / 5.8 = 0.55,
    43: 5.8 / 5.8 = 1.0,   # Best BM25 score
    44: 1.1 / 5.8 = 0.19,
}
```

**Step 2: Combine with weights** (40% BM25, 60% semantic)
```python
# Node 42
combined_42 = (0.4 × 0.55) + (0.6 × 0.92) = 0.22 + 0.55 = 0.77

# Node 43
combined_43 = (0.4 × 1.0) + (0.6 × 0.78) = 0.40 + 0.47 = 0.87  ⭐ Highest!

# Node 44
combined_44 = (0.4 × 0.19) + (0.6 × 0.0) = 0.08

# Node 45 (semantic only)
combined_45 = (0.4 × 0.0) + (0.6 × 0.85) = 0.51
```

**Step 3: Rank by combined score**
```python
[
    (43, 0.87),  # Best of both worlds! 🏆
    (42, 0.77),  # Good semantic, decent BM25
    (45, 0.51),  # Semantic only
    (44, 0.08),  # Poor match
]
```

**Result**: Node 43 wins because it has both exact keywords AND semantic relevance! 🎯

---

## Tuning Hybrid Weights 🎛️

### When to Adjust Weights

**More BM25 (e.g., 60% BM25 / 40% semantic)**:
- Code-heavy documentation
- Lots of technical jargon
- Users search for exact function names, parameters, config keys

**More Semantic (e.g., 30% BM25 / 70% semantic)**:
- Prose-heavy documentation
- Conceptual guides
- Users ask natural language questions

### Example: Adjusting Weights

```python
# In CLI or custom script
from hackathon.retrieval import HybridSearcher

searcher = HybridSearcher(db)

# Code documentation (favor exact terms)
results = searcher.hybrid_search(
    "configure pool_size",
    bm25_weight=0.6,
    semantic_weight=0.4
)

# User guides (favor concepts)
results = searcher.hybrid_search(
    "how to improve connection performance",
    bm25_weight=0.3,
    semantic_weight=0.7
)
```

---

## Performance Optimization 🚀

### Current Performance

| Operation | Latency (p50) | Latency (p95) |
|-----------|---------------|---------------|
| BM25 search | ~50ms | ~120ms |
| Semantic search (GPU) | ~80ms | ~150ms |
| Hybrid search (total) | ~150ms | ~300ms |

**Bottlenecks**:
1. BM25 index rebuild (every query) - O(n) corpus loading
2. Query embedding generation - GPU encoding
3. Sequential database fetches - N+1 queries

### Optimization Strategies

#### 1. Cache BM25 Index

**Current** (rebuild every query):
```python
def bm25_search(self, query: str, top_k: int = 10):
    bm25_indices = self.db.execute(select(BM25Index)).scalars().all()  # ❌ Slow
    # ... build index ...
```

**Optimized** (cache in memory):
```python
class HybridSearcher:
    def __init__(self, db: Session):
        self.db = db
        self._bm25_cache = None  # Cache corpus

    def _get_bm25_index(self):
        if self._bm25_cache is None:
            bm25_indices = self.db.execute(select(BM25Index)).scalars().all()
            # Build and cache
            self._bm25_cache = self._build_bm25_corpus(bm25_indices)
        return self._bm25_cache
```

**Speedup**: ~3-5x for repeated queries

#### 2. Batch Node Fetches

**Current** (N+1 queries):
```python
for node_id, score in sorted_node_ids:
    stmt = select(DocumentNode).where(DocumentNode.id == node_id)
    node = self.db.execute(stmt).scalar_one_or_none()  # ❌ Separate query
```

**Optimized** (single query):
```python
# Get all node IDs
node_ids = [node_id for node_id, _ in sorted_node_ids]

# Fetch all in one query
stmt = select(DocumentNode).where(DocumentNode.id.in_(node_ids))
nodes = {n.id: n for n in self.db.execute(stmt).scalars().all()}

# Build results maintaining order
results = [
    SearchResult(..., node=nodes[node_id])
    for node_id, score in sorted_node_ids
    if node_id in nodes
]
```

**Speedup**: ~2-3x for multi-result queries

#### 3. pgvector Index Tuning

**Current**:
```sql
CREATE INDEX embeddings_vector_idx ON embeddings USING hnsw (vector vector_cosine_ops);
```

**Optimized** (tune HNSW parameters):
```sql
CREATE INDEX embeddings_vector_idx ON embeddings
USING hnsw (vector vector_cosine_ops)
WITH (m = 16, ef_construction = 64);  -- Default: m=16, ef_construction=64

-- For queries, set ef_search
SET hnsw.ef_search = 40;  -- Default: 40, higher = more accurate but slower
```

**Trade-offs**:
- Higher `m`: Better recall, more memory
- Higher `ef_construction`: Better quality, slower indexing
- Higher `ef_search`: Better recall, slower queries

---

## CLI Usage 💻

### Interactive Query Mode

```bash
uv run query
```

```
🔍 RAG System Query Interface

Enter your query (or 'quit' to exit): postgres pooling configuration

Searching for: postgres pooling configuration

┌──────┬────────┬──────────┬────────────┬──────────────────────────────────┐
│ Rank │ Score  │ Type     │ Path       │ Preview                          │
├──────┼────────┼──────────┼────────────┼──────────────────────────────────┤
│ 1    │ 0.8734 │ code     │ chunk_42   │ pool = PoolManager(size=10)...   │
│ 2    │ 0.7821 │ paragraph│ chunk_15   │ Configure pool_size to 10 for... │
│ 3    │ 0.6543 │ list_item│ chunk_23   │ Set pool_pre_ping=True to...    │
└──────┴────────┴──────────┴────────────┴──────────────────────────────────┘

Enter your query (or 'quit' to exit): quit

Goodbye! 👋
```

### Direct Query

```bash
uv run query "postgres pooling configuration"
```

### With Context Expansion

```bash
uv run query "postgres pooling" --expand-context
```

---

## Next Steps 📚

- [Context Expansion](context-expansion.md) - How results are enriched with hierarchical context
- [Database Schema](../architecture/database.md) - Index design and query patterns
- [Ingestion Pipeline](ingestion.md) - How documents become searchable
