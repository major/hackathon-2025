"""Embedding generation using ROCm-accelerated PyTorch."""

import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from hackathon.config import get_settings
from hackathon.database.operations import create_embedding
from hackathon.models.schemas import EmbeddingCreate


class EmbeddingGenerator:
    """Generate embeddings using granite-embedding model with ROCm support."""

    def __init__(self) -> None:
        """Initialize the embedding generator with ROCm-accelerated model."""
        self.settings = get_settings()
        self.device = self.settings.embedding_device

        # Verify CUDA/ROCm is available
        if self.device == "cuda" and not torch.cuda.is_available():
            msg = "CUDA/ROCm is not available. Please ensure PyTorch with ROCm 6.3 is installed."
            raise RuntimeError(msg)

        # Load the model
        self.model = SentenceTransformer(
            self.settings.embedding_model, device=self.device, trust_remote_code=True
        )

        self.batch_size = self.settings.embedding_batch_size

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [emb.tolist() for emb in embeddings]

    def create_embedding_for_node(self, db: Session, node_id: int, text: str) -> None:
        """
        Create and store embedding for a document node.

        Args:
            db: Database session
            node_id: ID of the document node
            text: Text content to embed
        """
        vector = self.embed_text(text)

        embedding_data = EmbeddingCreate(
            node_id=node_id, vector=vector, model_name=self.settings.embedding_model
        )

        create_embedding(db, embedding_data)

    def batch_create_embeddings(
        self, db: Session, node_texts: list[tuple[int, str]], progress_callback=None
    ) -> None:
        """
        Create embeddings for multiple nodes in batches with progress tracking.

        Args:
            db: Database session
            node_texts: List of (node_id, text) tuples
            progress_callback: Optional callback function for progress updates
        """
        total_batches = (len(node_texts) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(0, len(node_texts), self.batch_size):
            batch = node_texts[batch_idx : batch_idx + self.batch_size]

            # Extract texts and node IDs
            node_ids = [nid for nid, _ in batch]
            texts = [text for _, text in batch]

            # Generate embeddings in batch
            vectors = self.embed_batch(texts)

            # Store embeddings
            for node_id, vector in zip(node_ids, vectors, strict=False):
                embedding_data = EmbeddingCreate(
                    node_id=node_id,
                    vector=vector,
                    model_name=self.settings.embedding_model,
                )
                create_embedding(db, embedding_data)

            # Commit after each batch
            db.commit()

            # Update progress if callback provided
            if progress_callback:
                progress_callback(batch_idx // self.batch_size + 1, total_batches)
