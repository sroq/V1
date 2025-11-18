"""
Document loader using the unstructured library.
Handles loading and parsing various document formats.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from unstructured.partition.auto import partition
from unstructured.documents.elements import Element

from .utils import validate_file_size


@dataclass
class DocumentMetadata:
    """Metadata for a loaded document."""

    file_path: str
    file_name: str
    file_type: str
    file_size: int
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    num_elements: int = 0
    num_pages: Optional[int] = None
    custom_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'file_path': self.file_path,
            'file_name': self.file_name,
            'file_type': self.file_type,
            'file_size': self.file_size,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'modified_at': self.modified_at.isoformat() if self.modified_at else None,
            'num_elements': self.num_elements,
            'num_pages': self.num_pages,
            'custom_metadata': self.custom_metadata or {}
        }


@dataclass
class LoadedDocument:
    """A loaded document with its elements and metadata."""

    elements: List[Element]
    metadata: DocumentMetadata
    raw_content: Optional[str] = None

    def __len__(self) -> int:
        """Return number of elements."""
        return len(self.elements)


class DocumentLoader:
    """
    Loads and parses documents using the unstructured library.
    """

    SUPPORTED_EXTENSIONS = {
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'doc',
        '.txt': 'text',
        '.md': 'markdown',
        '.html': 'html',
        '.htm': 'html',
    }

    def __init__(
        self,
        supported_extensions: Optional[List[str]] = None,
        max_file_size_mb: int = 50,
        extract_metadata: str = "rich"
    ):
        """
        Initialize document loader.

        Args:
            supported_extensions: List of file extensions to support
            max_file_size_mb: Maximum file size in MB
            extract_metadata: Metadata extraction level (basic, standard, rich)
        """
        self.supported_extensions = supported_extensions or list(self.SUPPORTED_EXTENSIONS.keys())
        self.max_file_size_mb = max_file_size_mb
        self.extract_metadata = extract_metadata
        self.logger = logging.getLogger(__name__)

    def load_document(self, file_path: Union[str, Path]) -> Optional[LoadedDocument]:
        """
        Load a single document.

        Args:
            file_path: Path to the document

        Returns:
            LoadedDocument object or None if loading fails
        """
        file_path = Path(file_path)

        # Validate file
        if not self._validate_file(file_path):
            return None

        try:
            # Parse document using unstructured
            elements = partition(
                filename=str(file_path),
                strategy="auto",
                include_metadata=True,
                include_page_breaks=True,
            )

            # Extract metadata
            metadata = self._extract_metadata(file_path, elements)

            # Optionally extract raw content
            raw_content = None
            if self.extract_metadata == "rich":
                raw_content = "\n".join([str(el) for el in elements])

            self.logger.info(
                f"Loaded {file_path.name}: {len(elements)} elements, "
                f"{metadata.num_pages or 'N/A'} pages"
            )

            return LoadedDocument(
                elements=elements,
                metadata=metadata,
                raw_content=raw_content
            )

        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}", exc_info=True)
            return None

    def load_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True
    ) -> List[LoadedDocument]:
        """
        Load all supported documents from a directory.

        Args:
            directory_path: Path to directory
            recursive: Whether to search recursively

        Returns:
            List of LoadedDocument objects
        """
        directory_path = Path(directory_path)

        if not directory_path.is_dir():
            self.logger.error(f"{directory_path} is not a directory")
            return []

        # Find all supported files
        files = self._find_files(directory_path, recursive)
        self.logger.info(f"Found {len(files)} supported files in {directory_path}")

        # Load documents
        documents = []
        for file_path in files:
            doc = self.load_document(file_path)
            if doc:
                documents.append(doc)

        return documents

    def _validate_file(self, file_path: Path) -> bool:
        """
        Validate that a file can be loaded.

        Args:
            file_path: Path to file

        Returns:
            True if valid, False otherwise
        """
        # Check file exists
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return False

        # Check extension
        if file_path.suffix.lower() not in self.supported_extensions:
            self.logger.debug(f"Unsupported file type: {file_path.suffix}")
            return False

        # Check file size
        if not validate_file_size(file_path, self.max_file_size_mb):
            return False

        return True

    def _find_files(self, directory: Path, recursive: bool = True) -> List[Path]:
        """
        Find all supported files in a directory.

        Args:
            directory: Directory to search
            recursive: Whether to search recursively

        Returns:
            List of file paths
        """
        files = []

        if recursive:
            for ext in self.supported_extensions:
                files.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in self.supported_extensions:
                files.extend(directory.glob(f"*{ext}"))

        return sorted(files)

    def _extract_metadata(
        self,
        file_path: Path,
        elements: List[Element]
    ) -> DocumentMetadata:
        """
        Extract metadata from file and elements.

        Args:
            file_path: Path to file
            elements: Parsed elements

        Returns:
            DocumentMetadata object
        """
        # Get file stats
        stats = file_path.stat()

        # Extract page numbers if available
        num_pages = None
        if elements:
            page_numbers = set()
            for element in elements:
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'page_number'):
                    page_num = element.metadata.page_number
                    if page_num is not None:
                        page_numbers.add(page_num)
            if page_numbers:
                num_pages = max(page_numbers)

        # Custom metadata based on extraction level
        custom_metadata = {}

        if self.extract_metadata in ["standard", "rich"]:
            custom_metadata.update({
                'created_timestamp': datetime.fromtimestamp(stats.st_ctime).isoformat(),
                'modified_timestamp': datetime.fromtimestamp(stats.st_mtime).isoformat(),
            })

        if self.extract_metadata == "rich":
            # Extract structural information
            structure = self._extract_structure(elements)
            custom_metadata['structure'] = structure

        return DocumentMetadata(
            file_path=str(file_path.absolute()),
            file_name=file_path.name,
            file_type=self.SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), 'unknown'),
            file_size=stats.st_size,
            created_at=datetime.fromtimestamp(stats.st_ctime),
            modified_at=datetime.fromtimestamp(stats.st_mtime),
            num_elements=len(elements),
            num_pages=num_pages,
            custom_metadata=custom_metadata
        )

    def _extract_structure(self, elements: List[Element]) -> Dict[str, Any]:
        """
        Extract document structure from elements.

        Args:
            elements: Document elements

        Returns:
            Structure information
        """
        structure = {
            'headings': [],
            'sections': 0,
            'paragraphs': 0,
            'tables': 0,
            'lists': 0,
        }

        current_section = None

        for element in elements:
            element_type = type(element).__name__

            # Extract headings
            if 'Title' in element_type:
                heading_text = str(element)
                level = 1  # Default level

                # Try to determine heading level
                if hasattr(element.metadata, 'category'):
                    if 'h1' in element.metadata.category.lower():
                        level = 1
                    elif 'h2' in element.metadata.category.lower():
                        level = 2
                    elif 'h3' in element.metadata.category.lower():
                        level = 3

                heading_info = {
                    'text': heading_text,
                    'level': level,
                }

                # Add page number if available
                if hasattr(element.metadata, 'page_number') and element.metadata.page_number:
                    heading_info['page'] = element.metadata.page_number

                structure['headings'].append(heading_info)

                if current_section != heading_text:
                    current_section = heading_text
                    structure['sections'] += 1

            # Count other elements
            elif 'Paragraph' in element_type or 'NarrativeText' in element_type:
                structure['paragraphs'] += 1
            elif 'Table' in element_type:
                structure['tables'] += 1
            elif 'List' in element_type:
                structure['lists'] += 1

        return structure
