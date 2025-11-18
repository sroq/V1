"""
Database uploader for PostgreSQL with pgvector.
Handles uploading chunks and embeddings to the database.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import hashlib

import psycopg
from psycopg.rows import dict_row
from psycopg import Connection

from .strategies import Chunk
from .embeddings import EmbeddingResult
from .loader import DocumentMetadata
from .utils import create_progress_bar


class DatabaseUploader:
    """
    Uploads document chunks and embeddings to PostgreSQL with pgvector.
    """

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        batch_size: int = 500,
        update_strategy: str = "replace",
    ):
        """
        Initialize database uploader.

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            batch_size: Number of records to insert per transaction
            update_strategy: How to handle updates (replace, version, upsert)
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.batch_size = batch_size
        self.update_strategy = update_strategy

        self.logger = logging.getLogger(__name__)
        self._connection: Optional[connection] = None

        self.logger.info(
            f"Initialized DatabaseUploader: {user}@{host}:{port}/{database}, "
            f"strategy={update_strategy}"
        )

    def connect(self) -> None:
        """Establish database connection."""
        try:
            self._connection = psycopg.connect(
                host=self.host,
                port=self.port,
                dbname=self.database,
                user=self.user,
                password=self.password,
            )
            self.logger.info("Successfully connected to database")

            # Verify pgvector extension
            self._verify_pgvector()

        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}", exc_info=True)
            raise

    def disconnect(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            self.logger.info("Disconnected from database")

    def _verify_pgvector(self) -> None:
        """Verify that pgvector extension is installed."""
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                )
                exists = cursor.fetchone()[0]

                if not exists:
                    raise RuntimeError(
                        "pgvector extension is not installed. "
                        "Run: CREATE EXTENSION vector;"
                    )

                self.logger.info("pgvector extension verified")

        except Exception as e:
            self.logger.error(f"Error verifying pgvector: {e}", exc_info=True)
            raise

    def upload_document(
        self,
        metadata: DocumentMetadata,
        chunks: List[Chunk],
        embeddings: List[EmbeddingResult],
        show_progress: bool = True
    ) -> Tuple[int, int]:
        """
        Upload a document with its chunks and embeddings.

        Args:
            metadata: Document metadata
            chunks: List of chunks
            embeddings: List of embeddings
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (document_id, num_chunks_uploaded)
        """
        if not self._connection:
            raise RuntimeError("Not connected to database")

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})"
            )

        try:
            # Insert document metadata
            document_id = self._insert_document(metadata)
            self.logger.info(f"Inserted document: {document_id}")

            # Upload chunks with embeddings
            num_uploaded = self._upload_chunks(
                document_id=document_id,
                chunks=chunks,
                embeddings=embeddings,
                show_progress=show_progress
            )

            self._connection.commit()
            self.logger.info(
                f"Successfully uploaded document {document_id} with {num_uploaded} chunks"
            )

            return document_id, num_uploaded

        except Exception as e:
            self._connection.rollback()
            self.logger.error(f"Error uploading document: {e}", exc_info=True)
            raise

    def _map_file_type_to_source_type(self, file_type: str) -> str:
        """
        Map file type to database source_type enum.

        Args:
            file_type: File type from metadata

        Returns:
            source_type enum value (pdf, docx, txt, md, html, url, api, other)
        """
        if not file_type:
            return 'other'

        # Normalize to lowercase
        file_type_lower = file_type.lower()

        # Map common variations
        type_map = {
            'markdown': 'md',
            'pdf': 'pdf',
            'docx': 'docx',
            'doc': 'docx',
            'txt': 'txt',
            'text': 'txt',
            'html': 'html',
            'htm': 'html',
            'md': 'md',
        }

        return type_map.get(file_type_lower, 'other')

    def _insert_document(self, metadata: DocumentMetadata) -> int:
        """
        Insert document metadata and return document ID.

        Args:
            metadata: Document metadata

        Returns:
            Document ID
        """
        # Handle update strategy
        if self.update_strategy == "replace":
            # Delete existing document with same path
            self._delete_document_by_path(metadata.file_path)

        with self._connection.cursor() as cursor:
            # Generate file hash from file path (simple hash for identification)
            file_hash = hashlib.sha256(metadata.file_path.encode()).hexdigest()

            # Insert into documents table
            # Schema: title, source_type, source_url, file_path, file_size_bytes, file_hash, metadata, processed_at
            cursor.execute(
                """
                INSERT INTO documents (
                    title, source_type, file_path, file_size_bytes,
                    file_hash, metadata, processed_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    metadata.file_name,  # title
                    self._map_file_type_to_source_type(metadata.file_type),  # source_type
                    metadata.file_path,
                    metadata.file_size,  # file_size_bytes
                    file_hash,  # file_hash
                    json.dumps(metadata.custom_metadata or {}),
                    datetime.now(),
                )
            )

            document_id = cursor.fetchone()[0]
            return document_id

    def _delete_document_by_path(self, file_path: str) -> None:
        """
        Delete existing document and its chunks by file path.

        Args:
            file_path: Path to document
        """
        with self._connection.cursor() as cursor:
            # Get document ID
            cursor.execute(
                "SELECT id FROM documents WHERE file_path = %s",
                (file_path,)
            )
            result = cursor.fetchone()

            if result:
                document_id = result[0]

                # Delete chunks (will cascade if foreign key is set up)
                cursor.execute(
                    "DELETE FROM document_chunks WHERE document_id = %s",
                    (document_id,)
                )

                # Delete document
                cursor.execute(
                    "DELETE FROM documents WHERE id = %s",
                    (document_id,)
                )

                self.logger.info(f"Deleted existing document: {file_path}")

    def _upload_chunks(
        self,
        document_id: int,
        chunks: List[Chunk],
        embeddings: List[EmbeddingResult],
        show_progress: bool = True
    ) -> int:
        """
        Upload chunks with embeddings in batches.

        Args:
            document_id: Document ID
            chunks: List of chunks
            embeddings: List of embeddings
            show_progress: Whether to show progress bar

        Returns:
            Number of chunks uploaded
        """
        num_uploaded = 0
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        # Create embedding lookup
        embedding_map = {emb.chunk_id: emb.embedding for emb in embeddings}

        for batch_idx in create_progress_bar(
            range(total_batches),
            desc="Uploading chunks",
            show=show_progress,
            unit="batch"
        ):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(chunks))
            batch = chunks[start_idx:end_idx]

            # Prepare batch data
            batch_data = []
            for chunk in batch:
                embedding = embedding_map.get(chunk.chunk_id)
                if not embedding:
                    self.logger.warning(f"Missing embedding for chunk {chunk.chunk_id}")
                    continue

                # Schema: id (auto), document_id, chunk_index, content, embedding, token_count, metadata, created_at (auto)
                batch_data.append((
                    document_id,
                    chunk.chunk_index,
                    chunk.content,
                    embedding,
                    chunk.token_count,
                    json.dumps(chunk.metadata),
                ))

            # Insert batch
            with self._connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO document_chunks (
                        document_id, chunk_index, content, embedding, token_count, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    batch_data
                )

            num_uploaded += len(batch_data)

            self.logger.debug(
                f"Batch {batch_idx + 1}/{total_batches}: Uploaded {len(batch_data)} chunks"
            )

        return num_uploaded

    def get_document_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata by file path.

        Args:
            file_path: Path to document

        Returns:
            Document metadata dict or None if not found
        """
        if not self._connection:
            raise RuntimeError("Not connected to database")

        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, file_path, file_name, file_type, file_size,
                       created_at, modified_at, metadata, processed_at
                FROM documents
                WHERE file_path = %s
                """,
                (file_path,)
            )

            row = cursor.fetchone()
            if not row:
                return None

            return {
                'id': row[0],
                'file_path': row[1],
                'file_name': row[2],
                'file_type': row[3],
                'file_size': row[4],
                'created_at': row[5],
                'modified_at': row[6],
                'metadata': row[7],
                'processed_at': row[8],
            }

    def get_document_chunks(self, document_id: int) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.

        Args:
            document_id: Document ID

        Returns:
            List of chunk dictionaries
        """
        if not self._connection:
            raise RuntimeError("Not connected to database")

        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, chunk_id, content, chunk_index, metadata,
                       token_count, content_hash, embedding
                FROM document_chunks
                WHERE document_id = %s
                ORDER BY chunk_index
                """,
                (document_id,)
            )

            chunks = []
            for row in cursor.fetchall():
                chunks.append({
                    'id': row[0],
                    'chunk_id': row[1],
                    'content': row[2],
                    'chunk_index': row[3],
                    'metadata': row[4],
                    'token_count': row[5],
                    'content_hash': row[6],
                    'embedding': row[7],
                })

            return chunks

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics
        """
        if not self._connection:
            raise RuntimeError("Not connected to database")

        with self._connection.cursor() as cursor:
            # Count documents
            cursor.execute("SELECT COUNT(*) FROM documents")
            num_documents = cursor.fetchone()[0]

            # Count chunks
            cursor.execute("SELECT COUNT(*) FROM document_chunks")
            num_chunks = cursor.fetchone()[0]

            # Total tokens
            cursor.execute("SELECT SUM(token_count) FROM document_chunks")
            total_tokens = cursor.fetchone()[0] or 0

            # Average tokens per chunk
            avg_tokens = total_tokens / num_chunks if num_chunks > 0 else 0

            return {
                'num_documents': num_documents,
                'num_chunks': num_chunks,
                'total_tokens': total_tokens,
                'avg_tokens_per_chunk': round(avg_tokens, 2),
            }

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
