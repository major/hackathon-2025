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
    node_type: str
    text_content: str | None = None
    is_leaf: bool = False
    node_path: str
    position: int | None = None
    metadata: dict[str, str | int | float | bool] | None = None


class MultiFieldBM25IndexCreate(BaseModel):
    """Schema for creating a multi-field BM25 index entry."""

    node_id: int
    full_text: str
    headings: str
    summary: str
    contextual_text: str = ""  # Optional contextual summary + text


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
    """Schema for expanded context with neighbor nodes."""

    node_id: int
    text_content: str
    node_type: str
    node_path: str
    neighbors_before: list["ExpandedContext"] = Field(default_factory=list)
    neighbors_after: list["ExpandedContext"] = Field(default_factory=list)
