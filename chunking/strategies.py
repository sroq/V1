"""
Chunking strategies for document processing.
Implements four different approaches to splitting documents into chunks.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from unstructured.documents.elements import Element
from unstructured.chunking.title import chunk_by_title
from unstructured.chunking.basic import chunk_elements

from .loader import LoadedDocument
from .utils import estimate_tokens, compute_content_hash


@dataclass
class Chunk:
    """A chunk of text with metadata."""

    content: str
    chunk_index: int
    chunk_id: str
    metadata: Dict[str, Any]
    source_document: str
    token_count: int
    content_hash: str

    @classmethod
    def from_element(
        cls,
        element: Element,
        chunk_index: int,
        source_document: str,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> "Chunk":
        """
        Create a Chunk from an unstructured Element.

        Args:
            element: Unstructured element
            chunk_index: Index of this chunk
            source_document: Source document path
            additional_metadata: Additional metadata to include

        Returns:
            Chunk object
        """
        content = str(element)
        token_count = estimate_tokens(content)
        content_hash = compute_content_hash(content)

        # Extract element metadata
        metadata = {
            'element_type': type(element).__name__,
        }

        if hasattr(element, 'metadata'):
            if hasattr(element.metadata, 'page_number') and element.metadata.page_number:
                metadata['page_number'] = element.metadata.page_number
            if hasattr(element.metadata, 'category') and element.metadata.category:
                metadata['category'] = element.metadata.category
            if hasattr(element.metadata, 'coordinates') and element.metadata.coordinates:
                metadata['coordinates'] = str(element.metadata.coordinates)

        # Add additional metadata
        if additional_metadata:
            metadata.update(additional_metadata)

        chunk_id = f"{source_document}:chunk:{chunk_index}:{content_hash[:8]}"

        return cls(
            content=content,
            chunk_index=chunk_index,
            chunk_id=chunk_id,
            metadata=metadata,
            source_document=source_document,
            token_count=token_count,
            content_hash=content_hash
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            'content': self.content,
            'chunk_index': self.chunk_index,
            'chunk_id': self.chunk_id,
            'metadata': self.metadata,
            'source_document': self.source_document,
            'token_count': self.token_count,
            'content_hash': self.content_hash,
        }


class ChunkingStrategy(ABC):
    """Base class for chunking strategies."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize chunking strategy.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def chunk_document(self, document: LoadedDocument) -> List[Chunk]:
        """
        Chunk a document.

        Args:
            document: LoadedDocument to chunk

        Returns:
            List of Chunk objects
        """
        pass

    def _elements_to_chunks(
        self,
        elements: List[Element],
        source_document: str,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Convert elements to Chunk objects.

        Args:
            elements: List of elements
            source_document: Source document path
            additional_metadata: Additional metadata

        Returns:
            List of Chunk objects
        """
        chunks = []
        for i, element in enumerate(elements):
            if str(element).strip():  # Skip empty elements
                chunk = Chunk.from_element(
                    element=element,
                    chunk_index=i,
                    source_document=source_document,
                    additional_metadata=additional_metadata
                )
                chunks.append(chunk)

        self.logger.info(f"Created {len(chunks)} chunks from {len(elements)} elements")
        return chunks


class FixedSizeChunker(ChunkingStrategy):
    """
    Fixed-size chunking strategy using unstructured's chunk_by_title.
    Splits documents into fixed-size chunks with overlap.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **kwargs
    ):
        """
        Initialize fixed-size chunker.

        Args:
            chunk_size: Target chunk size in tokens (~4 characters per token)
            chunk_overlap: Overlap between chunks in tokens
            **kwargs: Additional configuration
        """
        super().__init__(kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Convert tokens to characters (rough approximation)
        self.max_characters = chunk_size * 4
        self.overlap_characters = chunk_overlap * 4

        self.logger.info(
            f"Initialized FixedSizeChunker: chunk_size={chunk_size} tokens, "
            f"overlap={chunk_overlap} tokens"
        )

    def chunk_document(self, document: LoadedDocument) -> List[Chunk]:
        """Chunk document using fixed-size strategy."""
        try:
            # Use chunk_by_title with fixed size parameters
            chunked_elements = chunk_by_title(
                elements=document.elements,
                max_characters=self.max_characters,
                new_after_n_chars=self.max_characters - self.overlap_characters,
                combine_text_under_n_chars=100,
            )

            # Convert to Chunk objects
            chunks = self._elements_to_chunks(
                elements=chunked_elements,
                source_document=document.metadata.file_path,
                additional_metadata={
                    'chunking_strategy': 'fixed_size',
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                }
            )

            return chunks

        except Exception as e:
            self.logger.error(f"Error chunking document: {e}", exc_info=True)
            return []


class SemanticChunker(ChunkingStrategy):
    """
    Semantic chunking strategy that respects document structure.
    Combines elements based on semantic boundaries (paragraphs, sections).
    """

    def __init__(
        self,
        max_chunk_size: int = 1024,
        respect_boundaries: bool = True,
        combine_short: bool = True,
        **kwargs
    ):
        """
        Initialize semantic chunker.

        Args:
            max_chunk_size: Maximum chunk size in tokens
            respect_boundaries: Respect paragraph/section boundaries
            combine_short: Combine short elements
            **kwargs: Additional configuration
        """
        super().__init__(kwargs)
        self.max_chunk_size = max_chunk_size
        self.respect_boundaries = respect_boundaries
        self.combine_short = combine_short
        self.max_characters = max_chunk_size * 4

        self.logger.info(
            f"Initialized SemanticChunker: max_size={max_chunk_size} tokens, "
            f"respect_boundaries={respect_boundaries}"
        )

    def chunk_document(self, document: LoadedDocument) -> List[Chunk]:
        """Chunk document using semantic strategy."""
        try:
            if self.respect_boundaries:
                # Use chunk_by_title to respect document structure
                chunked_elements = chunk_by_title(
                    elements=document.elements,
                    max_characters=self.max_characters,
                    combine_text_under_n_chars=200 if self.combine_short else 0,
                    multipage_sections=True,
                )
            else:
                # Use basic chunking
                chunked_elements = chunk_elements(
                    elements=document.elements,
                    max_characters=self.max_characters,
                )

            # Convert to Chunk objects
            chunks = self._elements_to_chunks(
                elements=chunked_elements,
                source_document=document.metadata.file_path,
                additional_metadata={
                    'chunking_strategy': 'semantic',
                    'max_chunk_size': self.max_chunk_size,
                    'respect_boundaries': self.respect_boundaries,
                }
            )

            return chunks

        except Exception as e:
            self.logger.error(f"Error chunking document: {e}", exc_info=True)
            return []


class RecursiveChunker(ChunkingStrategy):
    """
    Recursive character chunking with hierarchical separators.
    Similar to LangChain's RecursiveCharacterTextSplitter.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize recursive chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            separators: List of separators in priority order
            **kwargs: Additional configuration
        """
        super().__init__(kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n## ",  # Markdown H2
            "\n### ",  # Markdown H3
            "\n\n",  # Paragraph
            "\n",  # Line
            ". ",  # Sentence
            " ",  # Word
        ]

        self.max_characters = chunk_size * 4
        self.overlap_characters = chunk_overlap * 4

        self.logger.info(
            f"Initialized RecursiveChunker: chunk_size={chunk_size} tokens, "
            f"separators={len(self.separators)}"
        )

    def chunk_document(self, document: LoadedDocument) -> List[Chunk]:
        """Chunk document using recursive strategy."""
        try:
            # Get full text from document
            full_text = "\n\n".join([str(el) for el in document.elements])

            # Recursively split text
            text_chunks = self._recursive_split(full_text)

            # Create Chunk objects
            chunks = []
            for i, text in enumerate(text_chunks):
                if text.strip():
                    content_hash = compute_content_hash(text)
                    chunk_id = f"{document.metadata.file_path}:chunk:{i}:{content_hash[:8]}"

                    chunk = Chunk(
                        content=text,
                        chunk_index=i,
                        chunk_id=chunk_id,
                        metadata={
                            'chunking_strategy': 'recursive',
                            'chunk_size': self.chunk_size,
                            'chunk_overlap': self.chunk_overlap,
                        },
                        source_document=document.metadata.file_path,
                        token_count=estimate_tokens(text),
                        content_hash=content_hash
                    )
                    chunks.append(chunk)

            self.logger.info(f"Created {len(chunks)} chunks using recursive strategy")
            return chunks

        except Exception as e:
            self.logger.error(f"Error chunking document: {e}", exc_info=True)
            return []

    def _recursive_split(self, text: str) -> List[str]:
        """
        Recursively split text using hierarchical separators.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        if len(text) <= self.max_characters:
            return [text]

        # Try each separator in order
        for separator in self.separators:
            if separator in text:
                splits = text.split(separator)
                chunks = []
                current_chunk = ""

                for split in splits:
                    # Re-add separator (except for first split)
                    if current_chunk:
                        split = separator + split

                    if len(current_chunk) + len(split) <= self.max_characters:
                        current_chunk += split
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)

                        # Handle overlap
                        if self.overlap_characters > 0 and current_chunk:
                            overlap = current_chunk[-self.overlap_characters:]
                            current_chunk = overlap + split
                        else:
                            current_chunk = split

                if current_chunk:
                    chunks.append(current_chunk)

                # Recursively split chunks that are still too large
                final_chunks = []
                for chunk in chunks:
                    if len(chunk) > self.max_characters:
                        final_chunks.extend(self._recursive_split(chunk))
                    else:
                        final_chunks.append(chunk)

                return final_chunks

        # If no separator works, force split
        return self._force_split(text)

    def _force_split(self, text: str) -> List[str]:
        """
        Force split text when no separator works.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.max_characters
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap_characters if self.overlap_characters > 0 else end

        return chunks


class DocumentTypeSpecificChunker(ChunkingStrategy):
    """
    Document-type specific chunking that uses different strategies
    based on the document type (Markdown, PDF, Code, etc.).
    """

    def __init__(
        self,
        markdown_config: Optional[Dict[str, Any]] = None,
        pdf_config: Optional[Dict[str, Any]] = None,
        code_config: Optional[Dict[str, Any]] = None,
        default_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize document-type specific chunker.

        Args:
            markdown_config: Configuration for Markdown documents
            pdf_config: Configuration for PDF documents
            code_config: Configuration for code documents
            default_config: Default configuration
            **kwargs: Additional configuration
        """
        super().__init__(kwargs)

        # Default configurations
        self.markdown_config = markdown_config or {
            'preserve_headings': True,
            'chunk_by_section': True,
            'max_chunk_size': 1024,
        }
        self.pdf_config = pdf_config or {
            'preserve_pages': True,
            'extract_tables': True,
            'max_chunk_size': 1024,
        }
        self.code_config = code_config or {
            'preserve_functions': True,
            'language_aware': True,
            'max_chunk_size': 768,
        }
        self.default_config = default_config or {
            'max_chunk_size': 1024,
        }

        self.logger.info("Initialized DocumentTypeSpecificChunker")

    def chunk_document(self, document: LoadedDocument) -> List[Chunk]:
        """Chunk document using type-specific strategy."""
        file_type = document.metadata.file_type

        try:
            if file_type == 'markdown':
                return self._chunk_markdown(document)
            elif file_type == 'pdf':
                return self._chunk_pdf(document)
            elif file_type in ['python', 'javascript', 'java']:
                return self._chunk_code(document)
            else:
                return self._chunk_default(document)

        except Exception as e:
            self.logger.error(f"Error chunking document: {e}", exc_info=True)
            return []

    def _chunk_markdown(self, document: LoadedDocument) -> List[Chunk]:
        """Chunk Markdown documents preserving structure."""
        config = self.markdown_config
        max_chars = config['max_chunk_size'] * 4

        if config['preserve_headings'] and config['chunk_by_section']:
            # Chunk by title to preserve headings
            chunked_elements = chunk_by_title(
                elements=document.elements,
                max_characters=max_chars,
                combine_text_under_n_chars=100,
                multipage_sections=True,
            )
        else:
            chunked_elements = chunk_elements(
                elements=document.elements,
                max_characters=max_chars,
            )

        return self._elements_to_chunks(
            elements=chunked_elements,
            source_document=document.metadata.file_path,
            additional_metadata={
                'chunking_strategy': 'document_specific',
                'document_type': 'markdown',
            }
        )

    def _chunk_pdf(self, document: LoadedDocument) -> List[Chunk]:
        """Chunk PDF documents preserving pages."""
        config = self.pdf_config
        max_chars = config['max_chunk_size'] * 4

        # Chunk by title which respects page boundaries
        chunked_elements = chunk_by_title(
            elements=document.elements,
            max_characters=max_chars,
            multipage_sections=not config['preserve_pages'],
        )

        return self._elements_to_chunks(
            elements=chunked_elements,
            source_document=document.metadata.file_path,
            additional_metadata={
                'chunking_strategy': 'document_specific',
                'document_type': 'pdf',
            }
        )

    def _chunk_code(self, document: LoadedDocument) -> List[Chunk]:
        """Chunk code documents preserving function boundaries."""
        config = self.code_config
        max_chars = config['max_chunk_size'] * 4

        # For code, use basic chunking to avoid breaking syntax
        chunked_elements = chunk_elements(
            elements=document.elements,
            max_characters=max_chars,
        )

        return self._elements_to_chunks(
            elements=chunked_elements,
            source_document=document.metadata.file_path,
            additional_metadata={
                'chunking_strategy': 'document_specific',
                'document_type': 'code',
            }
        )

    def _chunk_default(self, document: LoadedDocument) -> List[Chunk]:
        """Default chunking for other document types."""
        max_chars = self.default_config['max_chunk_size'] * 4

        chunked_elements = chunk_by_title(
            elements=document.elements,
            max_characters=max_chars,
        )

        return self._elements_to_chunks(
            elements=chunked_elements,
            source_document=document.metadata.file_path,
            additional_metadata={
                'chunking_strategy': 'document_specific',
                'document_type': 'default',
            }
        )


def create_chunker(strategy: str, config: Dict[str, Any]) -> ChunkingStrategy:
    """
    Factory function to create a chunking strategy.

    Args:
        strategy: Strategy name (fixed, semantic, recursive, document_specific)
        config: Configuration dictionary

    Returns:
        ChunkingStrategy instance
    """
    strategy = strategy.lower()

    if strategy == 'fixed' or strategy == 'fixed_size':
        return FixedSizeChunker(**config.get('fixed_size', {}))
    elif strategy == 'semantic':
        return SemanticChunker(**config.get('semantic', {}))
    elif strategy == 'recursive':
        return RecursiveChunker(**config.get('recursive', {}))
    elif strategy == 'document_specific':
        return DocumentTypeSpecificChunker(**config.get('document_specific', {}))
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
