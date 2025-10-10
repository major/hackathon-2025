"""Central constants for the RAG system.

This module contains all magic numbers and configuration defaults used throughout
the system. Centralizing these makes them easy to find, document, and tune.
"""

# ============================================================================
# BM25 Search Parameters 📊
# ============================================================================

# BM25 algorithm parameters (standard values from literature)
BM25_K1 = 1.5  # Term frequency saturation parameter
BM25_B = 0.75  # Length normalization parameter

# Reciprocal Rank Fusion parameter (k=60 is standard from literature)
BM25_RRF_K = 60

# Default number of search results
DEFAULT_TOP_K = 5

# ============================================================================
# Document Chunking Parameters 📄
# ============================================================================

# Docling HybridChunker settings
DEFAULT_CHUNK_SIZE = 512  # Maximum tokens per chunk
MAX_CHUNK_SIZE = 1024  # Absolute maximum chunk size

# Tokenizer for chunking (must be cached in ~/.cache/huggingface/)
CHUNKING_TOKENIZER = "bert-base-uncased"

# ============================================================================
# Reranking Parameters 🎯
# ============================================================================

# IBM Watsonx reranker model token limit
RERANKER_TOKEN_LIMIT = 512

# Conservative character limit to stay within token budget
# Accounts for: query tokens (50-100) + tokenization variance + safety margin
MAX_DOCUMENT_CHARS = 800  # ~200 tokens

# Default number of candidates to retrieve before reranking
DEFAULT_RERANK_CANDIDATES = 50

# Recommended range for reranking candidates
MIN_RERANK_CANDIDATES = 20
MAX_RERANK_CANDIDATES = 100

# ============================================================================
# Contextual Retrieval Parameters 🤖
# ============================================================================

# Common LLM response prefixes to strip (despite instructions not to add them)
LLM_RESPONSE_PREFIXES = [
    "Context:",
    "Summary:",
    "Description:",
    "This section",
    "This chunk",
]

# Maximum retries for LLM API calls
MAX_LLM_RETRIES = 3

# ============================================================================
# Display/UI Parameters 🖥️
# ============================================================================

# Text preview length in search results
PREVIEW_LENGTH = 200

# Default number of neighbor chunks to show
DEFAULT_NEIGHBORS_BEFORE = 1
DEFAULT_NEIGHBORS_AFTER = 1

# ============================================================================
# Database Parameters 🗄️
# ============================================================================

# PostgreSQL FTS language dictionary
POSTGRES_FTS_DICTIONARY = "english"

# ============================================================================
# Parallel Processing Parameters 🚀
# ============================================================================

# Number of concurrent threads for document processing
DOCUMENT_PROCESSING_THREADS = 4

# Number of concurrent threads for LLM API calls
LLM_API_THREADS = 10

# ============================================================================
# File Paths 📁
# ============================================================================

# Default BM25 index storage directory
BM25_INDEX_DIR = ".bm25s_index"

# Supported document extensions
SUPPORTED_DOCUMENT_EXTENSIONS = [".md", ".markdown"]

# ============================================================================
# Logging Parameters 📢
# ============================================================================

# Default log level for application code
DEFAULT_LOG_LEVEL = "WARNING"

# Libraries to silence (set to CRITICAL)
SILENCED_LIBRARIES = [
    "grpc",
    "transformers",
    "bm25s",
    "docling",
    "ibm_watsonx_ai",
]
