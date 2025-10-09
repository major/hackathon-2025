# Architecture Overview

This document provides a high-level overview of the RAG system architecture, explaining how components interact and why design decisions were made.

## System Goals

1. **Accurate Retrieval**: Find the most relevant document chunks for a given query
2. **Context Preservation**: Maintain document structure and relationships
3. **Hybrid Search**: Combine keyword matching with semantic similarity
4. **Scalability**: Handle 1K-10K documents efficiently

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG SYSTEM LAYERS                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐         ┌──────────────────┐           │
│  │   CLI Layer    │ ◄──────►│  Query Interface │           │
│  │  (process.py,  │         │    (query.py)    │           │
│  │   clean.py)    │         └──────────────────┘           │
│  └───────┬────────┘                                         │
│          │                                                  │
│          ▼                                                  │
│  ┌─────────────────────────────────────────────┐           │
│  │         Processing Layer                     │           │
│  │  ┌──────────────┐  ┌───────────────────┐   │           │
│  │  │   Docling    │  │  BM25 Indexing    │   │           │
│  │  │  Processor   │  │  Embedding Gen    │   │           │
│  │  │              │  │  Contextual Proc  │   │           │
│  │  └──────────────┘  └───────────────────┘   │           │
│  └──────────────────────┬──────────────────────┘           │
│                         │                                   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────┐          │
│  │         Data Layer (PostgreSQL)               │          │
│  │  ┌──────────┐  ┌─────────┐  ┌────────────┐  │          │
│  │  │Documents │  │  Nodes  │  │ BM25 Index │  │          │
│  │  └──────────┘  └─────────┘  └────────────┘  │          │
│  │  ┌──────────┐  ┌─────────┐                   │          │
│  │  │Embeddings│  │Contextual│                  │          │
│  │  │(pgvector)│  │  Chunks  │                  │          │
│  │  └──────────┘  └─────────┘                   │          │
│  └──────────────────────┬───────────────────────┘          │
│                         │                                   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────┐          │
│  │         Retrieval Layer                       │          │
│  │  ┌──────────────┐  ┌────────────────────┐   │          │
│  │  │ Hybrid       │  │ Context            │   │          │
│  │  │ Searcher     │  │ Expander           │   │          │
│  │  └──────────────┘  └────────────────────┘   │          │
│  └───────────────────────────────────────────────┘          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Document Processing Pipeline

**Purpose**: Transform markdown files into searchable, structured chunks

**Key Files**:
- `src/hackathon/processing/docling_processor.py`
- `src/hackathon/cli/process.py`

**Flow**:
```
Markdown → YAML Extract → Docling Parse → Chunk → Store Nodes
```

### 2. Indexing System

**Purpose**: Create multiple indexes for different search strategies

**Components**:
- **BM25 Index**: Keyword-based full-text search
- **Vector Index**: Semantic similarity (384-dim embeddings)
- **Contextual Index**: LLM-enhanced chunk context

### 3. Retrieval System

**Purpose**: Find and rank relevant document chunks

**Key Files**:
- `src/hackathon/retrieval/search.py` - Hybrid search
- `src/hackathon/retrieval/context_expansion.py` - Context building

**Search Strategy**:
```
Query → BM25 (40%) + Semantic (60%) → Merge & Rank → Expand Context
```

### 4. Database Layer

**Purpose**: Persist and query structured document data

**Technology**: PostgreSQL with pgvector extension

**Schema**:
```
documents (1) ──< (N) document_nodes
                         │
            ┌────────────┼────────────┐
            │            │            │
            ▼            ▼            ▼
       embeddings   bm25_index  contextual_chunks
```

## Data Flow: End-to-End

### Ingestion Flow

```
┌─────────────┐
│  blog/      │
│  *.md files │
└──────┬──────┘
       │
       │ 1. Read file
       ▼
┌─────────────────────────┐
│ python-frontmatter      │
│ Extract YAML metadata   │
│ Clean content           │
└────────┬────────────────┘
         │
         │ 2. Create temp file (no frontmatter)
         ▼
┌─────────────────────────┐
│ Docling Converter       │
│ Parse markdown structure│
│ Detect: headings, code, │
│   lists, tables, etc.   │
└────────┬────────────────┘
         │
         │ 3. Intelligent chunking
         ▼
┌─────────────────────────┐
│ Docling HybridChunker   │
│ - Respects boundaries   │
│ - Max token limit       │
│ - Preserves hierarchy   │
└────────┬────────────────┘
         │
         │ 4. For each chunk
         ▼
┌──────────────────────────────────────┐
│  Extract metadata:                   │
│  - doc_items (structure info)        │
│  - headings (context)                │
│  - node_type (paragraph/code/list)   │
│  - origin (source tracking)          │
└────────┬─────────────────────────────┘
         │
         │ 5. Store in PostgreSQL
         ▼
┌─────────────────────────────────────┐
│  Create DocumentNode                │
│  - document_id (foreign key)        │
│  - text_content                     │
│  - metadata (JSON)                  │
│  - is_leaf = True                   │
└────────┬────────────────────────────┘
         │
         │ 6. Index in parallel
         │
    ┌────┴────┬────────────┬──────────────┐
    │         │            │              │
    ▼         ▼            ▼              ▼
┌───────┐ ┌──────┐  ┌──────────┐  ┌──────────────┐
│ BM25  │ │Vector│  │Contextual│  │ (commit to   │
│Tokens │ │Embed │  │ LLM      │  │  database)   │
│       │ │(GPU) │  │ Summary  │  │              │
└───────┘ └──────┘  └──────────┘  └──────────────┘
```

### Query Flow

```
┌──────────────┐
│ User Query:  │
│ "How to X?"  │
└──────┬───────┘
       │
       │ 1. Parallel search
       │
    ┌──┴────────────────┐
    │                   │
    ▼                   ▼
┌─────────┐      ┌──────────────┐
│  BM25   │      │  Semantic    │
│ Search  │      │  Search      │
│         │      │  (pgvector)  │
│ SELECT  │      │  SELECT *    │
│  FROM   │      │  ORDER BY    │
│ bm25_idx│      │  embedding   │
│         │      │  <=> query   │
└────┬────┘      └──────┬───────┘
     │                  │
     │ 2. Merge results │
     │                  │
     └────────┬─────────┘
              │
              ▼
      ┌──────────────────┐
      │ Weighted Ranking │
      │ BM25:     40%    │
      │ Semantic: 60%    │
      └────────┬─────────┘
               │
               │ 3. Top-K results
               ▼
      ┌──────────────────────┐
      │  For top result:     │
      │  - Get semantic block│
      │  - Build hierarchy   │
      │  - Expand context    │
      └────────┬─────────────┘
               │
               ▼
      ┌──────────────────────┐
      │  Display:            │
      │  [Semantic Block]    │
      │  [Hierarchical Ctx]  │
      └──────────────────────┘
```

## Design Decisions

### Why Docling?

**Decision**: Use Docling instead of custom markdown parsing

**Rationale**:
- Intelligent chunking that respects semantic boundaries
- Rich metadata (headings, document items, origin tracking)
- Handles complex structures (nested lists, code blocks, tables)
- Industry-tested library from IBM Research

**Trade-off**: Requires temporary file creation to remove YAML frontmatter

### Why Hybrid Search?

**Decision**: Combine BM25 (40%) + Semantic (60%) instead of pure vector search

**Rationale**:
- BM25 excels at exact keyword matches (e.g., technical terms, code)
- Semantic search handles conceptual queries better
- Hybrid approach covers both use cases

**Implementation**:
```python
bm25_score = bm25_results[node_id] * 0.4
semantic_score = semantic_results[node_id] * 0.6
final_score = bm25_score + semantic_score
```

### Why Contextual Retrieval?

**Decision**: Generate LLM summaries of each chunk with surrounding context

**Rationale**:
- Chunks alone lack context (e.g., "Option 2: Configure X" is meaningless without knowing what "X" refers to)
- LLM can synthesize context from ancestors
- Improves retrieval accuracy per Anthropic research

**Cost**: ~$0.001-0.01 per document (Claude API calls)

### Why PostgreSQL + pgvector?

**Decision**: Single database for all data instead of separate vector DB

**Rationale**:
- Simplifies deployment (one database)
- ACID transactions for consistency
- pgvector is production-ready for <1M vectors
- Familiar SQL interface for developers

**Alternative considered**: Pinecone, Weaviate (overkill for POC)

## Scalability Considerations

### Current Limits

| Component | Limit | Bottleneck |
|-----------|-------|------------|
| Document count | ~10K | BM25 in-memory rebuild |
| Query latency | ~500ms | Sequential DB queries |
| Embedding generation | ~5 docs/sec | GPU throughput |
| Contextual processing | ~2 docs/sec | LLM API rate limits |

### Scaling Strategies

**For 10K-100K documents**:
1. Replace rank-bm25 with Tantivy or PostgreSQL FTS
2. Cache BM25 index in Redis
3. Batch LLM calls more aggressively
4. Add query result caching

**For 100K+ documents**:
1. Shard by document collection
2. Move to dedicated vector DB (Pinecone, Qdrant)
3. Pre-compute common query embeddings
4. Distribute embedding generation

## Error Handling Strategy

### Processing Pipeline
- **Docling failure**: Skip document, log error, continue
- **GPU unavailable**: Fall back to CPU (with warning)
- **LLM timeout**: Store empty contextual chunk, continue

### Query Path
- **No results**: Return empty list (valid state)
- **DB connection lost**: Raise exception, let CLI handle
- **Malformed query**: Return validation error

## Monitoring & Observability

### Logging Strategy
```python
# Structured logging with rich
logger.info("Processing document", extra={
    "document_id": doc_id,
    "chunks": len(chunks),
    "duration_ms": elapsed
})
```

### Key Metrics to Track
- **Ingestion rate**: docs/second
- **Index size**: MB per document type
- **Query latency**: p50, p95, p99
- **Result relevance**: click-through rate (if UI exists)

## Security Considerations

### Data Privacy
- No PII in logs
- Sanitize user queries before LLM calls
- Database credentials via environment variables only

### Input Validation
- Markdown file size limits (prevent DoS)
- Query length limits
- SQL injection prevention (SQLAlchemy ORM)

## Future Enhancements

### Planned
1. **Reranking**: Add cross-encoder for result re-ranking
2. **Chunking strategies**: Support multiple chunk sizes
3. **Multi-modal**: Handle images, PDFs
4. **Real-time updates**: Incremental indexing without full rebuild

### Experimental
1. **Graph RAG**: Build knowledge graph from documents
2. **Query expansion**: Use LLM to expand user queries
3. **Active learning**: Learn from user feedback

## References

- [Anthropic Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
- [Docling Documentation](https://github.com/DS4SD/docling)
- [pgvector Best Practices](https://github.com/pgvector/pgvector)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
