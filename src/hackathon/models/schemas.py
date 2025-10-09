"""Pydantic schemas for data validation and transfer."""

from pydantic import BaseModel, ConfigDict, Field


class DocumentCreate(BaseModel):
    """Schema for creating a new document."""

    filename: str
    filepath: str
    metadata: dict[str, str | int | float | bool] | None = None


class DocumentNodeCreate(BaseModel):
    """Schema for creating a document node."""

    document_id: int
    parent_id: int | None = None
    node_type: str
    text_content: str | None = None
    is_leaf: bool = False
    node_path: str
    metadata: dict[str, str | int | float | bool] | None = None


class EmbeddingCreate(BaseModel):
    """Schema for creating an embedding."""

    node_id: int
    vector: list[float]
    model_name: str


class BM25IndexCreate(BaseModel):
    """Schema for creating BM25 index data."""

    node_id: int
    tokens: list[str]
    token_count: int


class ContextualChunkCreate(BaseModel):
    """Schema for creating a contextual chunk."""

    node_id: int
    original_text: str
    contextual_summary: str
    contextualized_text: str


class SearchResult(BaseModel):
    """Schema for search results."""

    model_config = ConfigDict(from_attributes=True)

    node_id: int
    document_id: int
    text_content: str
    node_type: str
    node_path: str
    score: float
    context: str | None = None
    metadata: dict[str, str | int | float | bool] | None = None


class ExpandedContext(BaseModel):
    """Schema for expanded context with parent nodes."""

    node_id: int
    text_content: str
    node_type: str
    node_path: str
    parents: list["ExpandedContext"] = Field(default_factory=list)
    children: list["ExpandedContext"] = Field(default_factory=list)
