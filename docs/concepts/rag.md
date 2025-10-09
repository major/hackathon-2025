# RAG (Retrieval-Augmented Generation) Explained

This document explains RAG concepts for developers familiar with databases and APIs but new to AI/ML systems.

## What is RAG?

**RAG** = **Retrieval** + **Augmented** + **Generation**

Think of it as a smart documentation search system that:
1. **Retrieves** relevant chunks from a knowledge base
2. **Augments** an LLM's context with retrieved information
3. **Generates** answers based on retrieved facts (not in this POC - we stop at retrieval)

## Traditional Search vs. RAG

### Traditional Search (like grep or PostgreSQL FTS)

```
User query: "connection pooling"
           ↓
    Keyword matching
           ↓
    Results containing
    exact phrase
    "connection pooling"
```

**Limitations**:
- Misses synonyms ("connection pool", "database pooling")
- Misses related concepts ("connection management", "pool configuration")
- No ranking by relevance/meaning

### RAG Search (Hybrid)

```
User query: "connection pooling"
           ↓
    ┌──────┴──────┐
    ▼             ▼
 Keywords      Meaning
 (BM25)        (Vectors)
    │             │
    ▼             ▼
 "connection"  [...semantic...]
 "pooling"     [..embedding...]
    │             │
    └──────┬──────┘
           ▼
    Merge & Rank
           ▼
    Results about:
    - "connection pool"
    - "database pooling"
    - "pool_size config"
    - Related concepts
```

**Advantages**:
- Finds semantic matches (meaning-based)
- Handles synonyms and paraphrases
- Ranks by relevance, not just keyword frequency

## Core Components

### 1. Document Chunking

**Problem**: LLMs and embedding models have token limits (~512 tokens for most models).

**Solution**: Split documents into smaller chunks that fit within limits.

```
Original Document (5000 tokens)
           ↓
    [Chunk 1: 400 tokens]
    [Chunk 2: 380 tokens]
    [Chunk 3: 420 tokens]
    [Chunk 4: 390 tokens]
    ...
```

**Challenge**: Where to split?

❌ **Naive approach**: Every N characters
```
"... database connection pooling is important. It
reduces overhead by reu"  ← BAD: cuts mid-sentence
```

✅ **Smart approach** (Docling): Respect semantic boundaries
```
"... database connection pooling is important. It
reduces overhead by reusing connections."

--- NEW CHUNK ---

"Connection pool configuration requires setting..."
```

### 2. Vector Embeddings

**Concept**: Convert text to numbers that capture meaning.

**Analogy**: Like GPS coordinates for concepts
- Words with similar meaning → Close coordinates
- Unrelated words → Far apart

```python
# Simplified example (real embeddings are 384-dim)
"database pooling"     → [0.8, 0.3, -0.1, ...]
"connection pool"      → [0.75, 0.35, -0.05, ...]  # Similar!
"cooking recipe"       → [-0.2, 0.9, 0.7, ...]     # Different!

# Distance calculation
cosine_similarity(embedding1, embedding2) → 0.95 (very similar)
cosine_similarity(embedding1, embedding3) → 0.12 (unrelated)
```

**How it works**:
1. Train a neural network on massive text corpus
2. Network learns to encode meaning in vector space
3. Similar concepts → Similar vectors

**In our system**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('ibm-granite/granite-embedding-30m-english')
embedding = model.encode("database connection pooling")
# → numpy array of shape (384,)
```

### 3. Hybrid Search

Combining two search strategies for better results.

#### BM25 (Keyword Search)

**Algorithm**: Okapi BM25 - Enhanced TF-IDF

**Formula** (simplified):
```
score = IDF(term) × (TF(term) × (k1 + 1)) / (TF(term) + k1 × (1 - b + b × doc_length/avg_length))

Where:
- IDF: Inverse Document Frequency (rarity of term)
- TF: Term Frequency (how often term appears in doc)
- k1: Term frequency saturation (default: 1.5)
- b: Length normalization (default: 0.75)
```

**Example**:
```
Query: "python pooling"

Document 1: "Python database connection pooling guide"
- TF(python) = 1, IDF(python) = 2.5 → score_python = 2.1
- TF(pooling) = 1, IDF(pooling) = 3.8 → score_pooling = 3.2
- Total BM25 score = 5.3

Document 2: "Java connection pooling"
- TF(python) = 0 → score_python = 0
- TF(pooling) = 1, IDF(pooling) = 3.8 → score_pooling = 3.2
- Total BM25 score = 3.2

Result: Document 1 ranks higher
```

**Strengths**:
- ✅ Exact keyword matches
- ✅ Fast (especially with proper indexing)
- ✅ Handles technical terms well

**Weaknesses**:
- ❌ Misses synonyms
- ❌ No semantic understanding
- ❌ Sensitive to vocabulary mismatch

#### Vector Search (Semantic)

**Algorithm**: Approximate Nearest Neighbor (ANN) with cosine similarity

**Process**:
```
1. Encode query to vector
   "python pooling" → [0.8, 0.3, -0.1, ..., 0.5]

2. Find nearest neighbors in vector space
   SELECT embedding FROM embeddings
   ORDER BY embedding <=> query_vector  -- cosine distance
   LIMIT 10;

3. Retrieve documents for top matches
```

**Example**:
```
Query: "python pooling"
Query vector: [0.8, 0.3, -0.1, ...]

Document 1: "Python database connection pooling guide"
Embedding: [0.75, 0.35, -0.05, ...]
Similarity: 0.95  ← High!

Document 2: "Database connection management in Python"
Embedding: [0.78, 0.28, -0.08, ...]
Similarity: 0.89  ← Also high! (even without "pooling")

Document 3: "Java multithreading tutorial"
Embedding: [0.1, -0.5, 0.8, ...]
Similarity: 0.23  ← Low
```

**Strengths**:
- ✅ Semantic understanding
- ✅ Handles synonyms and paraphrases
- ✅ Finds conceptually related content

**Weaknesses**:
- ❌ May miss exact technical terms
- ❌ Slower than keyword search
- ❌ Requires GPU for encoding

#### Hybrid = BM25 + Vector

**Why combine both?**

Different strengths for different query types:

| Query Type | Best Strategy |
|------------|---------------|
| "configure pool_size parameter" | BM25 (exact term) |
| "how to manage database connections" | Vector (semantic) |
| "pool_size configuration guide" | **Hybrid** (both!) |

**Implementation**:
```python
# Weight scores from each method
bm25_score = bm25_results[doc_id] * 0.4   # 40% weight
semantic_score = semantic_results[doc_id] * 0.6  # 60% weight
final_score = bm25_score + semantic_score
```

**Example**:
```
Query: "database pooling best practices"

Document A:
- BM25 score: 0.85 (has "database", "pooling", "practices")
- Semantic score: 0.70 (good semantic match)
- Final: 0.85×0.4 + 0.70×0.6 = 0.76

Document B:
- BM25 score: 0.60 (only has "database")
- Semantic score: 0.95 (excellent semantic match!)
- Final: 0.60×0.4 + 0.95×0.6 = 0.81  ← Winner!
```

### 4. Contextual Retrieval

**Problem**: Chunks lose context

```
Chunk text: "Set this to 10 for optimal performance"
User sees: "What should I set to 10??" 😕
```

**Solution**: Use LLM to add context

```
Original chunk:
"Set pool_size to 10 for optimal performance"

Context from ancestors:
"Document about PostgreSQL tuning.
Section: Connection Pool Configuration"

LLM prompt:
"Given this context: <context>
Summarize this chunk: <chunk>
In 1-2 sentences."

LLM output:
"In PostgreSQL connection pool configuration,
set pool_size parameter to 10 for optimal performance."
```

**Storage**:
```sql
-- Original chunk
INSERT INTO document_nodes (text_content) VALUES
('Set pool_size to 10 for optimal performance');

-- LLM-enhanced context
INSERT INTO contextual_chunks (node_id, contextual_summary) VALUES
(42, 'In PostgreSQL connection pool configuration, set pool_size parameter to 10 for optimal performance');
```

**Search benefits**:
- Query: "postgres pool size recommendation"
- Matches contextual chunk (has "PostgreSQL", "pool", "size")
- Original chunk might not match well (lacks "PostgreSQL")

## RAG vs. Fine-Tuning

### Fine-Tuning an LLM

```
Collect training data → Train model → Deploy new model

Pros:
+ Model "memorizes" knowledge
+ Fast inference (no retrieval)

Cons:
- Expensive ($1000s for training)
- Slow to update (retrain for new data)
- May hallucinate outdated info
```

### RAG System

```
Index documents → Retrieve relevant chunks → Augment LLM context

Pros:
+ Cheap (just indexing + retrieval)
+ Easy to update (add new docs)
+ Traceable (see source chunks)
+ Less hallucination (grounded in sources)

Cons:
- Slower (retrieval + generation)
- Quality depends on retrieval accuracy
```

**When to use RAG**:
- ✅ Frequently updated knowledge base
- ✅ Need source attribution
- ✅ Budget constraints
- ✅ Multiple knowledge domains

**When to fine-tune**:
- ✅ Fixed knowledge domain
- ✅ Need fastest inference
- ✅ Have large training budget
- ✅ Custom writing style needed

## This System's RAG Pipeline

### Ingestion

```
Markdown files
    ↓
Extract YAML frontmatter (python-frontmatter)
    ↓
Parse structure (Docling)
    ↓
Intelligent chunking (Docling HybridChunker)
    ↓
Store chunks (PostgreSQL)
    ↓
Parallel indexing:
├── BM25: Tokenize terms → bm25_index table
├── Vectors: Encode with granite → embeddings table (pgvector)
└── Context: LLM summaries → contextual_chunks table
```

### Retrieval

```
User query
    ↓
Parallel search:
├── BM25: Load index → Tokenize query → Score with BM25Okapi
└── Vector: Encode query → pgvector ANN search → Cosine similarity
    ↓
Merge results (40% BM25, 60% Vector)
    ↓
Rank by combined score
    ↓
Context expansion:
├── Reassemble semantic blocks (lists, code, tables)
└── Build hierarchical context (ancestors, siblings)
    ↓
Return enriched results
```

## Key Metrics

### Retrieval Quality

**Recall@K**: Did we retrieve relevant chunks in top K?
```
recall@5 = (relevant chunks in top 5) / (total relevant chunks)
```

**Precision@K**: How many top K chunks are relevant?
```
precision@5 = (relevant chunks in top 5) / 5
```

**MRR (Mean Reciprocal Rank)**: How high is first relevant result?
```
MRR = 1 / (rank of first relevant chunk)
```

### Speed Metrics

- **Indexing speed**: docs/second
- **Query latency**: ms per query (p50, p95, p99)
- **Embedding time**: ms per chunk

### Cost Metrics

- **Storage**: MB per 1000 docs
- **Compute**: GPU hours for embedding
- **LLM costs**: $ per 1000 contextual summaries

## Common Pitfalls

### 1. Chunking Too Large/Small

❌ **Too large** (>1000 tokens):
- Won't fit in embedding model
- Returns irrelevant content mixed with relevant

❌ **Too small** (<50 tokens):
- Loses context
- Fragments semantic meaning

✅ **Just right** (200-500 tokens):
- Fits in models
- Preserves semantic units

### 2. Ignoring Chunk Boundaries

❌ **Bad chunking**:
```
Chunk 1: "...connection pooling is\n```python\npool = Pool"
Chunk 2: "Manager(size=10)\n```\nThis configures..."
```
Code split across chunks!

✅ **Good chunking** (Docling):
```
Chunk 1: "...connection pooling is essential."
Chunk 2: "```python\npool = PoolManager(size=10)\n```\nThis configures..."
```
Complete code block in one chunk.

### 3. Not Tuning Hybrid Weights

Default 40% BM25 / 60% Vector may not fit your data.

**Experiment**:
```python
# Code-heavy docs: Favor BM25 (exact terms matter)
weights = {'bm25': 0.6, 'semantic': 0.4}

# Prose docs: Favor semantic
weights = {'bm25': 0.3, 'semantic': 0.7}
```

### 4. Skipping Contextual Retrieval

Without context augmentation:
```
Chunk: "Run this command with --verbose flag"
Query: "how to enable verbose mode in tool X"
Result: ❌ No match (missing "tool X")
```

With contextual retrieval:
```
Chunk: "Run this command with --verbose flag"
Context: "Documentation for tool X, CLI options section"
Enhanced: "In tool X CLI options, run this command with --verbose flag"
Query: "how to enable verbose mode in tool X"
Result: ✅ Match!
```

## Further Reading

- [Anthropic's Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
- [BM25 Algorithm Explained](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Sentence Transformers](https://www.sbert.net/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
