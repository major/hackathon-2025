# Context Expansion

This document explains how the RAG system enriches search results by reconstructing semantic blocks and building hierarchical context from the document tree structure.

## Problem Statement 🤔

**Without context expansion**:
```
Search result: "Set pool_size to 10 for optimal performance"

User reaction: "What is pool_size? What system are we talking about?" 😕
```

**With context expansion**:
```
[Hierarchical Context]
PostgreSQL Guide > Connection Pooling > Configuration

[Semantic Block - Complete Code Example]
from psycopg2.pool import ThreadedConnectionPool

pool = ThreadedConnectionPool(
    minconn=1,
    maxconn=10,
    pool_size=10,  # ← This is what was matched!
    dsn="postgresql://localhost/mydb"
)

[Current Chunk]
Set pool_size to 10 for optimal performance

User reaction: "Perfect! Now I know exactly what to do!" ✅
```

## Architecture 🏗️

```
┌────────────────────────┐
│   Search Result        │
│   (DocumentNode)       │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────────────────┐
│   ContextExpander                  │
│   ┌──────────────────────────────┐ │
│   │ 1. Check for semantic block  │ │
│   │    (code, list, table)       │ │
│   └────────────┬─────────────────┘ │
│                │                    │
│                ├─ Has code? ───────────► Reassemble code block
│                ├─ Has list? ───────────► Reassemble list
│                └─ Has table? ──────────► Reassemble table
│                                     │
│   ┌──────────────────────────────┐ │
│   │ 2. Build hierarchical tree   │ │
│   │    - Traverse ancestors      │ │
│   │    - Include siblings (opt)  │ │
│   │    - Add current node (opt)  │ │
│   └──────────────────────────────┘ │
└────────────────┬───────────────────┘
                 │
                 ▼
      ┌────────────────────┐
      │ Enriched Context   │
      │ - Semantic block   │
      │ - Hierarchical path│
      │ - Full context     │
      └────────────────────┘
```

## Semantic Block Reassembly 🧩

### What are Semantic Blocks?

Semantic blocks are multi-chunk structures that Docling sometimes splits across multiple nodes:

- **Code blocks** 💻: Long functions, configuration files
- **Lists** 📝: Multi-item lists (ordered/unordered)
- **Tables** 📊: Multi-row tables

**Example Problem**:

**Original Markdown**:
```python
def configure_pool(size, timeout):
    """Configure connection pool with settings."""
    pool = ThreadedConnectionPool(
        minconn=1,
        maxconn=size,
        timeout=timeout,
        pool_pre_ping=True,
        pool_recycle=3600
    )
    return pool
```

**Docling Chunking** (might split at ~512 tokens):
```
Chunk 1:
def configure_pool(size, timeout):
    """Configure connection pool with settings."""
    pool = ThreadedConnectionPool(
        minconn=1,
        maxconn=size,

Chunk 2:
        timeout=timeout,
        pool_pre_ping=True,
        pool_recycle=3600
    )
    return pool
```

**Search returns Chunk 1**: User sees incomplete code! ❌

**With reassembly**: User sees complete function! ✅

### Implementation

**File**: `src/hackathon/retrieval/context_expansion.py:get_semantic_block_text()`

```python
def get_semantic_block_text(self, node: DocumentNode) -> str | None:
    """
    Attempt to reassemble semantic blocks (code, lists, tables).

    Returns full block text if node is part of a larger semantic structure,
    otherwise returns None.
    """
    metadata = node.meta or {}

    # Check if this chunk contains code based on Docling metadata
    node_type = node.node_type
    docling_types = metadata.get("docling_types", "")

    # Only try to reassemble if this node contains code
    if node_type != "code" and "code" not in docling_types:
        return None

    # Try to find adjacent chunks that are part of the same code block
    return self._reassemble_docling_code_block(node)
```

### Code Block Reassembly

**Strategy**: Find all sibling nodes with:
1. Same parent node
2. Same heading context
3. "code" in their `docling_types`

```python
def _reassemble_docling_code_block(self, node: DocumentNode) -> str:
    """Reassemble code blocks split across chunks."""
    metadata = node.meta or {}
    heading = metadata.get("headings", "")

    # Find all code chunks under same heading and parent
    all_nodes = get_all_leaf_nodes(self.db, node.document_id)

    related_chunks = []
    for other_node in all_nodes:
        if self._is_related_code_chunk(other_node, heading, node.parent_id):
            related_chunks.append(other_node)

    # Sort by path (maintains document order)
    related_chunks.sort(key=lambda n: n.node_path)

    # Join with newlines (preserve code formatting)
    return "\n".join(chunk.text_content or "" for chunk in related_chunks)

def _is_related_code_chunk(
    self, other_node: DocumentNode, heading: str, parent_id: int | None
) -> bool:
    """Check if a node is part of the same code block."""
    other_meta = other_node.meta or {}
    other_heading = other_meta.get("headings", "")
    other_docling_types = other_meta.get("docling_types", "")

    return (
        other_heading == heading
        and other_node.parent_id == parent_id
        and "code" in other_docling_types
    )
```

### Example: Code Block Reassembly

**Database State**:

```python
# Node 42 (chunk_0)
{
    "text_content": "def configure_pool(size, timeout):\n    pool = ThreadedConnectionPool(\n        minconn=1,",
    "node_type": "code",
    "parent_id": 10,
    "metadata": {
        "headings": "Database Guide, Connection Pooling",
        "docling_types": "code"
    }
}

# Node 43 (chunk_1)
{
    "text_content": "        maxconn=size,\n        timeout=timeout\n    )\n    return pool",
    "node_type": "code",
    "parent_id": 10,
    "metadata": {
        "headings": "Database Guide, Connection Pooling",
        "docling_types": "code"
    }
}
```

**Search returns**: Node 42

**get_semantic_block_text(node_42)** process:
1. ✅ Check: node_type == "code"
2. Find siblings: Same parent (10), same heading, contains "code"
3. Found: [Node 42, Node 43]
4. Sort by path: [chunk_0, chunk_1]
5. Join: `node_42.text + "\n" + node_43.text`

**Result**:
```python
def configure_pool(size, timeout):
    pool = ThreadedConnectionPool(
        minconn=1,
        maxconn=size,
        timeout=timeout
    )
    return pool
```

Complete code! 🎉

---

## Hierarchical Context Building 🌳

### Document Tree Structure

Documents are stored as hierarchical trees:

```
Document: "Database Guide"
│
├─ Section: "Connection Pooling"  (parent_id=NULL)
│  │
│  ├─ Paragraph: "Pooling reduces overhead..." (parent_id=1, chunk_0)
│  ├─ List: "Benefits:" (parent_id=1)
│  │  ├─ Item: "Faster connections" (parent_id=3, chunk_1)
│  │  └─ Item: "Better resource use" (parent_id=3, chunk_2)
│  │
│  └─ Code: "pool = Pool(...)" (parent_id=1, chunk_3)
│
└─ Section: "Query Optimization" (parent_id=NULL)
   └─ Paragraph: "Use indexes..." (parent_id=5, chunk_4)
```

### Building Context Text

**File**: `src/hackathon/retrieval/context_expansion.py:build_context_text()`

```python
def build_context_text(
    self,
    node: DocumentNode,
    depth: int = 1,
    include_siblings: bool = False,
    exclude_current: bool = False,
) -> str:
    """
    Build hierarchical context by walking up the tree.

    Args:
        node: Starting node
        depth: How many ancestor levels to include
        include_siblings: Include sibling nodes
        exclude_current: Don't include the current node (useful when semantic block is shown)

    Returns:
        Formatted context text
    """
    parts = []

    # Get ancestors (parents, grandparents, etc.)
    ancestors = get_node_ancestors(self.db, node.id)[:depth]

    # Add ancestors (reverse order: top-down)
    for ancestor in reversed(ancestors):
        if ancestor.text_content:
            parts.append(f"[{ancestor.node_type}] {ancestor.text_content}")

    # Add siblings if requested
    if include_siblings and node.parent_id:
        siblings = get_node_children(self.db, node.parent_id)
        for sibling in siblings:
            if sibling.id != node.id and sibling.text_content:
                parts.append(f"[sibling: {sibling.node_type}] {sibling.text_content}")

    # Add the node itself (unless excluded)
    if not exclude_current and node.text_content:
        parts.append(f"[current: {node.node_type}] {node.text_content}")

    return "\n\n".join(parts)
```

### Example: Hierarchical Context

**Query**: `"pool configuration"`

**Search returns**: Node (chunk_3) - Code block

**Tree**:
```
Section (id=1): "Connection Pooling"
├─ Paragraph (id=2, chunk_0): "Pooling reduces overhead..."
└─ Code (id=4, chunk_3): "pool = Pool(...)"  ← Search result
```

**build_context_text(node_4, depth=2)**:

```python
ancestors = get_node_ancestors(db, node_4.id)[:2]
# Returns: [Section(id=1)]

reversed(ancestors) = [Section(id=1)]

parts = [
    "[section_header] Connection Pooling",
    "[current: code] pool = Pool(...)"
]

context = "[section_header] Connection Pooling\n\n[current: code] pool = Pool(...)"
```

**User sees**:
```
Connection Pooling

pool = Pool(...)
```

Much clearer! 🎯

### Example: With Siblings

**build_context_text(node_4, depth=2, include_siblings=True)**:

```python
ancestors = [Section(id=1)]

siblings = get_node_children(db, parent_id=1)
# Returns: [Paragraph(id=2), Code(id=4)]

parts = [
    "[section_header] Connection Pooling",
    "[sibling: paragraph] Pooling reduces overhead...",
    "[current: code] pool = Pool(...)"
]
```

**User sees**:
```
Connection Pooling

Pooling reduces overhead...

pool = Pool(...)
```

Even more context! 📚

---

## Combined: Semantic Block + Hierarchical Context

The CLI (`cli/query.py`) combines both strategies:

```python
def _display_expanded_context(top_result, expander):
    """Display expanded context for the top search result."""
    db = next(get_db())

    try:
        # Fetch the node
        stmt = select(DocumentNode).where(DocumentNode.id == top_result.node_id)
        node = db.execute(stmt).scalar_one_or_none()

        if node:
            # Try to get semantic block (code/list/table reassembly)
            semantic_block_text = expander.get_semantic_block_text(node)
            has_expanded_block = semantic_block_text and semantic_block_text != node.text_content

            # Build hierarchical context
            # If we have a semantic block, exclude current node (already shown in block)
            context_text = expander.build_context_text(
                node,
                depth=2,
                include_siblings=False,
                exclude_current=has_expanded_block
            )

            # Display combined context
            if has_expanded_block:
                full_context = f"[Semantic Block]\n{semantic_block_text}\n\n[Hierarchical Context]\n{context_text}"
                console.print(Panel(full_context, title="Full Context (with Semantic Block)", border_style="green"))
            else:
                console.print(Panel(context_text, title="Full Context", border_style="green"))
    finally:
        db.close()
```

### Full Example

**Query**: `"pool size configuration"`

**Search returns**: Node 42 (chunk_1 of code block)

**Step 1: Semantic block reassembly**
```python
semantic_block_text = expander.get_semantic_block_text(node_42)
# Returns complete code block (reassembled from chunks 0-2)
```

**Result**:
```python
from psycopg2.pool import ThreadedConnectionPool

pool = ThreadedConnectionPool(
    minconn=1,
    maxconn=10,
    pool_size=10,
    dsn="postgresql://localhost/mydb"
)
```

**Step 2: Hierarchical context** (exclude_current=True because shown in semantic block)
```python
context_text = expander.build_context_text(node_42, depth=2, exclude_current=True)
```

**Result**:
```
[section_header] PostgreSQL Guide > Connection Pooling

[paragraph] Connection pooling reduces connection overhead by reusing established connections.
```

**Step 3: Combined display**
```
╭──────────────────── Full Context (with Semantic Block) ────────────────────╮
│ [Semantic Block]                                                           │
│ from psycopg2.pool import ThreadedConnectionPool                           │
│                                                                             │
│ pool = ThreadedConnectionPool(                                             │
│     minconn=1,                                                             │
│     maxconn=10,                                                            │
│     pool_size=10,                                                          │
│     dsn="postgresql://localhost/mydb"                                      │
│ )                                                                           │
│                                                                             │
│ [Hierarchical Context]                                                     │
│ [section_header] PostgreSQL Guide > Connection Pooling                     │
│                                                                             │
│ [paragraph] Connection pooling reduces connection overhead by reusing      │
│ established connections.                                                   │
╰─────────────────────────────────────────────────────────────────────────────╯
```

Perfect context! The user now understands:
- ✅ What system (PostgreSQL)
- ✅ What topic (Connection Pooling)
- ✅ Complete code example
- ✅ Why it matters

---

## API Reference 📚

### ContextExpander Class

```python
from hackathon.retrieval import ContextExpander

expander = ContextExpander(db)
```

### get_semantic_block_text()

**Purpose**: Reassemble code blocks, lists, or tables split across chunks.

**Signature**:
```python
def get_semantic_block_text(self, node: DocumentNode) -> str | None
```

**Returns**:
- `str`: Reassembled full block if node is part of a semantic structure
- `None`: If node is not part of a code/list/table or already complete

**Example**:
```python
# Node contains partial code
node = db.query(DocumentNode).get(42)

full_code = expander.get_semantic_block_text(node)
# → Complete code block with all chunks joined
```

### build_context_text()

**Purpose**: Build hierarchical context by traversing the document tree.

**Signature**:
```python
def build_context_text(
    self,
    node: DocumentNode,
    depth: int = 1,
    include_siblings: bool = False,
    exclude_current: bool = False,
) -> str
```

**Parameters**:
- `node`: Starting document node
- `depth`: Number of ancestor levels to include (1 = parent only, 2 = parent + grandparent, etc.)
- `include_siblings`: Whether to include sibling nodes (default: False)
- `exclude_current`: Whether to exclude the current node from context (useful when semantic block is shown separately)

**Returns**: Formatted context string with ancestor/sibling/current node text

**Example**:
```python
# Get context with ancestors up to 2 levels
context = expander.build_context_text(node, depth=2)

# Get context with siblings
context = expander.build_context_text(node, depth=1, include_siblings=True)

# Get context excluding current (when showing semantic block separately)
context = expander.build_context_text(node, depth=2, exclude_current=True)
```

### get_parent_section()

**Purpose**: Find the nearest section header ancestor.

**Signature**:
```python
def get_parent_section(self, node: DocumentNode) -> DocumentNode | None
```

**Returns**: First ancestor with `node_type == "section_header"`, or None

**Example**:
```python
section = expander.get_parent_section(node)
if section:
    print(f"Section: {section.text_content}")
```

---

## Performance Considerations ⚡

### Database Queries

Context expansion requires multiple queries:

1. **get_node_ancestors()**: Recursive parent lookup
2. **get_node_children()**: Sibling fetch
3. **get_all_leaf_nodes()**: All chunks for semantic block reassembly

**Optimization**: Use SQLAlchemy's `joinedload()` to prefetch relationships:

```python
from sqlalchemy.orm import joinedload

stmt = (
    select(DocumentNode)
    .where(DocumentNode.id == node_id)
    .options(joinedload(DocumentNode.parent))  # Prefetch parent
)
```

### Caching Strategies

For repeated context expansion (e.g., showing multiple results):

```python
class ContextExpander:
    def __init__(self, db: Session):
        self.db = db
        self._ancestor_cache = {}  # Cache ancestor chains

    def get_node_ancestors(self, node_id: int) -> list[DocumentNode]:
        if node_id in self._ancestor_cache:
            return self._ancestor_cache[node_id]

        ancestors = get_node_ancestors(self.db, node_id)
        self._ancestor_cache[node_id] = ancestors
        return ancestors
```

---

## CLI Usage 💻

### Basic Search (No Context)

```bash
uv run query "pool configuration"
```

Returns search results table only.

### With Context Expansion

```bash
uv run query "pool configuration" --expand-context
```

Returns search results + full context panel for top result.

### Interactive Mode with Context

```bash
uv run query
```

```
Enter your query (or 'quit' to exit): pool configuration

[Search results table...]

╭──────────────────── Full Context (with Semantic Block) ────────────────────╮
│ [Semantic Block]                                                           │
│ pool = ThreadedConnectionPool(...)                                         │
│                                                                             │
│ [Hierarchical Context]                                                     │
│ PostgreSQL Guide > Connection Pooling                                      │
╰─────────────────────────────────────────────────────────────────────────────╯
```

---

## Troubleshooting 🔧

### Code Block Not Reassembled

**Symptom**: Partial code shown despite `get_semantic_block_text()` being called

**Diagnosis**:
```python
# Check metadata
node = db.query(DocumentNode).get(42)
print(node.meta)
# Expected: {"docling_types": "code, ..."}
```

**Possible causes**:
1. ❌ Missing "code" in `docling_types`
2. ❌ Different headings for split chunks
3. ❌ Different parent IDs

**Fix**: Check ingestion pipeline metadata extraction

### Missing Hierarchical Context

**Symptom**: Only current node shown, no ancestors

**Diagnosis**:
```python
# Check parent relationship
node = db.query(DocumentNode).get(42)
print(f"Parent ID: {node.parent_id}")

if node.parent_id:
    parent = db.query(DocumentNode).get(node.parent_id)
    print(f"Parent text: {parent.text_content}")
```

**Possible causes**:
1. ❌ `parent_id` is NULL (root node)
2. ❌ Parent has no `text_content`

**Fix**: Verify document tree structure was built correctly during ingestion

---

## Next Steps 📖

- [Search System](search.md) - How results are found and ranked
- [Ingestion Pipeline](ingestion.md) - How tree structure is built
- [Development Guide](../development/usage.md) - Build custom context expansion logic
