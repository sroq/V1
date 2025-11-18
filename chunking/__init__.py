"""
RAG Assistant Chunking Pipeline

A comprehensive document chunking system for RAG-based AI assistants.
Supports multiple chunking strategies and integrates with PostgreSQL/pgvector.
"""

__version__ = "1.0.0"
__author__ = "RAG Assistant Team"

from .loader import DocumentLoader
from .strategies import (
    ChunkingStrategy,
    FixedSizeChunker,
    SemanticChunker,
    RecursiveChunker,
    DocumentTypeSpecificChunker,
)
from .embeddings import EmbeddingGenerator
from .database import DatabaseUploader

__all__ = [
    "DocumentLoader",
    "ChunkingStrategy",
    "FixedSizeChunker",
    "SemanticChunker",
    "RecursiveChunker",
    "DocumentTypeSpecificChunker",
    "EmbeddingGenerator",
    "DatabaseUploader",
]
