"""
Embedding generation using OpenAI's API.
Handles batch processing and retry logic for generating embeddings.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .strategies import Chunk
from .utils import create_progress_bar


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""

    chunk_id: str
    embedding: List[float]
    model: str
    dimensions: int


class EmbeddingGenerator:
    """
    Generates embeddings for text chunks using OpenAI's API.
    Includes batch processing and retry logic.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 100,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        """
        Initialize embedding generator.

        Args:
            api_key: OpenAI API key
            model: Embedding model name
            dimensions: Embedding dimensions
            batch_size: Number of texts to embed per API call
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=api_key,
            timeout=timeout,
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Initialized EmbeddingGenerator: model={model}, "
            f"dimensions={dimensions}, batch_size={batch_size}"
        )

    def generate_embeddings(
        self,
        chunks: List[Chunk],
        show_progress: bool = True
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for a list of chunks.

        Args:
            chunks: List of Chunk objects
            show_progress: Whether to show progress bar

        Returns:
            List of EmbeddingResult objects
        """
        if not chunks:
            self.logger.warning("No chunks provided for embedding generation")
            return []

        self.logger.info(f"Generating embeddings for {len(chunks)} chunks")

        results = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        # Process in batches
        for batch_idx in create_progress_bar(
            range(total_batches),
            desc="Generating embeddings",
            show=show_progress,
            unit="batch"
        ):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(chunks))
            batch = chunks[start_idx:end_idx]

            try:
                batch_results = self._embed_batch(batch)
                results.extend(batch_results)

                self.logger.debug(
                    f"Batch {batch_idx + 1}/{total_batches}: "
                    f"Generated {len(batch_results)} embeddings"
                )

            except Exception as e:
                self.logger.error(
                    f"Error processing batch {batch_idx + 1}/{total_batches}: {e}",
                    exc_info=True
                )
                # Continue with next batch instead of failing completely
                continue

        self.logger.info(f"Successfully generated {len(results)} embeddings")
        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
        )),
        reraise=True,
    )
    def _embed_batch(self, batch: List[Chunk]) -> List[EmbeddingResult]:
        """
        Embed a batch of chunks with retry logic.

        Args:
            batch: List of Chunk objects

        Returns:
            List of EmbeddingResult objects
        """
        # Extract texts
        texts = [chunk.content for chunk in batch]

        try:
            # Call OpenAI API
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
            )

            # Process response
            results = []
            for i, embedding_data in enumerate(response.data):
                result = EmbeddingResult(
                    chunk_id=batch[i].chunk_id,
                    embedding=embedding_data.embedding,
                    model=self.model,
                    dimensions=len(embedding_data.embedding),
                )
                results.append(result)

            return results

        except openai.RateLimitError as e:
            self.logger.warning(f"Rate limit hit, retrying: {e}")
            raise
        except openai.APIConnectionError as e:
            self.logger.warning(f"API connection error, retrying: {e}")
            raise
        except openai.APITimeoutError as e:
            self.logger.warning(f"API timeout, retrying: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during embedding: {e}", exc_info=True)
            raise

    def generate_single_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if failed
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text],
                dimensions=self.dimensions,
            )

            return response.data[0].embedding

        except Exception as e:
            self.logger.error(f"Error generating single embedding: {e}", exc_info=True)
            return None

    def estimate_cost(self, num_chunks: int) -> Dict[str, float]:
        """
        Estimate the cost of generating embeddings.

        Args:
            num_chunks: Number of chunks to embed

        Returns:
            Dictionary with cost estimates
        """
        # Pricing for text-embedding-3-small: $0.02 per 1M tokens
        # Rough estimate: average chunk is 400 tokens
        avg_tokens_per_chunk = 400
        total_tokens = num_chunks * avg_tokens_per_chunk

        cost_per_million = 0.02
        estimated_cost = (total_tokens / 1_000_000) * cost_per_million

        return {
            'num_chunks': num_chunks,
            'estimated_tokens': total_tokens,
            'estimated_cost_usd': round(estimated_cost, 4),
            'model': self.model,
        }

    def validate_api_key(self) -> bool:
        """
        Validate that the API key works.

        Returns:
            True if valid, False otherwise
        """
        try:
            # Try to generate a single embedding
            response = self.client.embeddings.create(
                model=self.model,
                input=["test"],
                dimensions=self.dimensions,
            )

            self.logger.info("API key validation successful")
            return True

        except openai.AuthenticationError:
            self.logger.error("API key validation failed: Invalid API key")
            return False
        except Exception as e:
            self.logger.error(f"API key validation failed: {e}")
            return False


class EmbeddingCache:
    """
    Simple in-memory cache for embeddings to avoid redundant API calls.
    """

    def __init__(self):
        """Initialize embedding cache."""
        self._cache: Dict[str, List[float]] = {}
        self.logger = logging.getLogger(__name__)

    def get(self, content_hash: str) -> Optional[List[float]]:
        """
        Get embedding from cache.

        Args:
            content_hash: Content hash

        Returns:
            Embedding vector or None if not cached
        """
        return self._cache.get(content_hash)

    def set(self, content_hash: str, embedding: List[float]) -> None:
        """
        Store embedding in cache.

        Args:
            content_hash: Content hash
            embedding: Embedding vector
        """
        self._cache[content_hash] = embedding

    def has(self, content_hash: str) -> bool:
        """
        Check if embedding is cached.

        Args:
            content_hash: Content hash

        Returns:
            True if cached, False otherwise
        """
        return content_hash in self._cache

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self.logger.info("Embedding cache cleared")

    def size(self) -> int:
        """
        Get number of cached embeddings.

        Returns:
            Cache size
        """
        return len(self._cache)

    def cache_rate(self, total_requests: int) -> float:
        """
        Calculate cache hit rate.

        Args:
            total_requests: Total number of requests

        Returns:
            Cache hit rate (0.0 to 1.0)
        """
        if total_requests == 0:
            return 0.0
        return self.size() / total_requests
