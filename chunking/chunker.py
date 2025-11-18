#!/usr/bin/env python3
"""
Main CLI entry point for the RAG Assistant Chunking Pipeline.

Processes documents, generates chunks using various strategies,
creates embeddings, and uploads to PostgreSQL/pgvector database.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import os

from dotenv import load_dotenv

from .loader import DocumentLoader, LoadedDocument
from .strategies import create_chunker, Chunk
from .embeddings import EmbeddingGenerator, EmbeddingResult
from .database import DatabaseUploader
from .utils import (
    setup_logging,
    load_config,
    merge_config,
    ProgressTracker,
    ProcessingStats,
)


class ChunkingPipeline:
    """
    Main chunking pipeline orchestrator.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize chunking pipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self._init_components()

    def _init_components(self) -> None:
        """Initialize pipeline components."""
        # Document loader
        doc_config = self.config['document_loading']
        self.loader = DocumentLoader(
            supported_extensions=doc_config['supported_extensions'],
            max_file_size_mb=doc_config['max_file_size_mb'],
            extract_metadata=self.config['metadata']['level'],
        )

        # Chunking strategy
        chunking_config = self.config['chunking']
        strategy_name = chunking_config['default_strategy']
        self.chunker = create_chunker(strategy_name, chunking_config)

        # Embedding generator
        emb_config = self.config['embeddings']
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.embedder = EmbeddingGenerator(
            api_key=api_key,
            model=emb_config['model'],
            dimensions=emb_config['dimensions'],
            batch_size=emb_config['batch_size'],
            max_retries=emb_config['max_retries'],
            timeout=emb_config['timeout'],
        )

        # Database uploader (optional)
        self.uploader: Optional[DatabaseUploader] = None
        if self.config.get('upload', False):
            db_config = self.config['database']
            self.uploader = DatabaseUploader(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password'],
                batch_size=db_config['batch_size'],
                update_strategy=db_config['update_strategy'],
            )

        # Progress tracker
        proc_config = self.config['processing']
        progress_file = None
        if proc_config['save_progress']:
            progress_file = Path(proc_config['progress_file'])

        self.tracker = ProgressTracker(
            progress_file=progress_file,
            enabled=proc_config['save_progress']
        )

        self.show_progress = proc_config['progress_bar']

        self.logger.info("Pipeline components initialized")

    def process_input(self, input_path: Path) -> None:
        """
        Process input (file or directory).

        Args:
            input_path: Path to file or directory
        """
        if input_path.is_file():
            # Process single file
            self.logger.info(f"Processing single file: {input_path}")
            documents = [self.loader.load_document(input_path)]
            documents = [doc for doc in documents if doc is not None]
        elif input_path.is_dir():
            # Process directory
            self.logger.info(f"Processing directory: {input_path}")
            recursive = self.config['document_loading']['recursive']
            documents = self.loader.load_directory(input_path, recursive=recursive)
        else:
            raise ValueError(f"Input path does not exist: {input_path}")

        if not documents:
            self.logger.warning("No documents loaded")
            return

        # Update stats
        self.tracker.stats.total_files = len(documents)
        self.logger.info(f"Loaded {len(documents)} documents")

        # Process documents
        self._process_documents(documents)

        # Finalize and print summary
        self.tracker.stats.finalize()
        self.tracker.stats.print_summary()

    def _process_documents(self, documents: List[LoadedDocument]) -> None:
        """
        Process a list of documents.

        Args:
            documents: List of LoadedDocument objects
        """
        for doc in documents:
            file_path = doc.metadata.file_path

            # Check if already processed
            if self.tracker.is_processed(file_path):
                self.logger.info(f"Skipping already processed file: {file_path}")
                continue

            try:
                self.logger.info(f"Processing: {file_path}")

                # Chunk document
                chunks = self.chunker.chunk_document(doc)
                if not chunks:
                    self.logger.warning(f"No chunks generated for {file_path}")
                    self.tracker.mark_processed(file_path, success=False)
                    self.tracker.stats.add_error(file_path, "No chunks generated")
                    continue

                self.logger.info(f"Generated {len(chunks)} chunks")
                self.tracker.update_stats(total_chunks=len(chunks))

                # Generate embeddings
                embeddings = self.embedder.generate_embeddings(
                    chunks,
                    show_progress=self.show_progress
                )
                self.tracker.update_stats(total_embeddings=len(embeddings))

                # Upload to database
                if self.uploader:
                    if not self.uploader._connection:
                        self.uploader.connect()

                    document_id, num_uploaded = self.uploader.upload_document(
                        metadata=doc.metadata,
                        chunks=chunks,
                        embeddings=embeddings,
                        show_progress=self.show_progress
                    )
                    self.tracker.update_stats(total_uploaded=num_uploaded)
                    self.logger.info(
                        f"Uploaded document {document_id} with {num_uploaded} chunks"
                    )

                # Mark as processed
                self.tracker.mark_processed(file_path, success=True)

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}", exc_info=True)
                self.tracker.mark_processed(file_path, success=False)
                self.tracker.stats.add_error(file_path, str(e))

        # Disconnect from database
        if self.uploader and self.uploader._connection:
            self.uploader.disconnect()

    def validate_setup(self) -> bool:
        """
        Validate that all components are properly configured.

        Returns:
            True if valid, False otherwise
        """
        self.logger.info("Validating pipeline setup...")

        # Validate OpenAI API key
        if not self.embedder.validate_api_key():
            self.logger.error("OpenAI API key validation failed")
            return False

        # Validate database connection if uploading
        if self.uploader:
            try:
                self.uploader.connect()
                self.uploader.disconnect()
                self.logger.info("Database connection validated")
            except Exception as e:
                self.logger.error(f"Database connection failed: {e}")
                return False

        self.logger.info("Pipeline setup validated successfully")
        return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG Assistant Chunking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file with semantic chunking
  python chunking/chunker.py --input document.pdf --strategy semantic

  # Process directory with fixed-size chunking and upload
  python chunking/chunker.py --input docs/ --strategy fixed --upload

  # Use custom config file
  python chunking/chunker.py --input docs/ --config my_config.yaml

  # Override chunk size and upload
  python chunking/chunker.py --input docs/ --chunk-size 1024 --upload
        """
    )

    # Required arguments
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input file or directory path'
    )

    # Optional arguments
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).parent / 'config.yaml',
        help='Configuration file path (default: config.yaml)'
    )

    parser.add_argument(
        '--strategy',
        choices=['fixed', 'semantic', 'recursive', 'document_specific'],
        help='Chunking strategy (overrides config)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        help='Chunk size in tokens (for fixed/recursive strategies)'
    )

    parser.add_argument(
        '--chunk-overlap',
        type=int,
        help='Chunk overlap in tokens (for fixed/recursive strategies)'
    )

    parser.add_argument(
        '--upload',
        action='store_true',
        help='Upload chunks to database'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for embedding generation'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate setup and exit'
    )

    parser.add_argument(
        '--clear-progress',
        action='store_true',
        help='Clear saved progress before starting'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (overrides config)'
    )

    parser.add_argument(
        '--env-file',
        type=Path,
        default=Path.cwd() / '.env',
        help='Path to .env file (default: .env)'
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load environment variables
    if args.env_file.exists():
        load_dotenv(args.env_file)
    else:
        load_dotenv()  # Try default locations

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    # Override config with command-line arguments
    if args.strategy:
        config['chunking']['default_strategy'] = args.strategy

    if args.chunk_size:
        config['chunking']['fixed_size']['chunk_size'] = args.chunk_size
        config['chunking']['recursive']['chunk_size'] = args.chunk_size

    if args.chunk_overlap:
        config['chunking']['fixed_size']['chunk_overlap'] = args.chunk_overlap
        config['chunking']['recursive']['chunk_overlap'] = args.chunk_overlap

    if args.batch_size:
        config['embeddings']['batch_size'] = args.batch_size

    if args.upload:
        config['upload'] = True

    if args.log_level:
        config['logging']['level'] = args.log_level

    # Setup logging
    log_config = config['logging']
    log_file = Path(log_config['file']) if log_config.get('file') else None
    setup_logging(
        level=log_config['level'],
        log_file=log_file,
        console=log_config['console']
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("RAG Assistant Chunking Pipeline")
    logger.info("=" * 60)

    try:
        # Initialize pipeline
        pipeline = ChunkingPipeline(config)

        # Clear progress if requested
        if args.clear_progress:
            pipeline.tracker.clear()
            logger.info("Cleared saved progress")

        # Validate setup if requested
        if args.validate:
            logger.info("Running validation only...")
            if pipeline.validate_setup():
                logger.info("Validation successful!")
                return 0
            else:
                logger.error("Validation failed!")
                return 1

        # Validate input path
        if not args.input.exists():
            logger.error(f"Input path does not exist: {args.input}")
            return 1

        # Process input
        pipeline.process_input(args.input)

        logger.info("Pipeline completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
