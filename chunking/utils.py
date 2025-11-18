"""
Utility functions for logging, progress tracking, and error handling.
"""

import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from tqdm import tqdm


@dataclass
class ProcessingStats:
    """Statistics for document processing."""

    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    total_uploaded: int = 0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    errors: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.start_time is None:
            self.start_time = datetime.now().isoformat()

    def add_error(self, file_path: str, error: str) -> None:
        """Add an error to the stats."""
        self.errors.append({
            "file": file_path,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })

    def finalize(self) -> None:
        """Mark processing as complete."""
        self.end_time = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def print_summary(self) -> None:
        """Print a summary of the processing statistics."""
        print("\n" + "=" * 60)
        print("CHUNKING PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Files processed: {self.processed_files}/{self.total_files}")
        print(f"Files failed: {self.failed_files}")
        print(f"Total chunks generated: {self.total_chunks}")
        print(f"Total embeddings created: {self.total_embeddings}")
        print(f"Total chunks uploaded: {self.total_uploaded}")

        if self.start_time and self.end_time:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            duration = end - start
            print(f"Processing time: {duration}")

        if self.errors:
            print(f"\nErrors encountered: {len(self.errors)}")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"  - {error['file']}: {error['error']}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more")

        print("=" * 60 + "\n")


class ProgressTracker:
    """Track and persist processing progress."""

    def __init__(self, progress_file: Optional[Path] = None, enabled: bool = True):
        """
        Initialize progress tracker.

        Args:
            progress_file: Path to save progress
            enabled: Whether to save progress to disk
        """
        self.progress_file = progress_file
        self.enabled = enabled
        self.processed_files: set[str] = set()
        self.stats = ProcessingStats()

        if self.enabled and self.progress_file and self.progress_file.exists():
            self._load_progress()

    def _load_progress(self) -> None:
        """Load progress from file."""
        try:
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                self.processed_files = set(data.get('processed_files', []))
                stats_data = data.get('stats', {})
                # Restore stats
                self.stats.total_files = stats_data.get('total_files', 0)
                self.stats.processed_files = stats_data.get('processed_files', 0)
                self.stats.failed_files = stats_data.get('failed_files', 0)
                self.stats.total_chunks = stats_data.get('total_chunks', 0)
                self.stats.total_embeddings = stats_data.get('total_embeddings', 0)
                self.stats.total_uploaded = stats_data.get('total_uploaded', 0)
                logging.info(f"Loaded progress: {len(self.processed_files)} files already processed")
        except Exception as e:
            logging.warning(f"Could not load progress file: {e}")

    def _save_progress(self) -> None:
        """Save progress to file."""
        if not self.enabled or not self.progress_file:
            return

        try:
            data = {
                'processed_files': list(self.processed_files),
                'stats': self.stats.to_dict(),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.progress_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.warning(f"Could not save progress file: {e}")

    def is_processed(self, file_path: str) -> bool:
        """Check if a file has been processed."""
        return file_path in self.processed_files

    def mark_processed(self, file_path: str, success: bool = True) -> None:
        """Mark a file as processed."""
        self.processed_files.add(file_path)
        if success:
            self.stats.processed_files += 1
        else:
            self.stats.failed_files += 1
        self._save_progress()

    def update_stats(self, **kwargs) -> None:
        """Update statistics."""
        for key, value in kwargs.items():
            if hasattr(self.stats, key):
                if isinstance(value, int):
                    # Add to existing value
                    current = getattr(self.stats, key)
                    setattr(self.stats, key, current + value)
                else:
                    setattr(self.stats, key, value)
        self._save_progress()

    def clear(self) -> None:
        """Clear progress."""
        if self.progress_file and self.progress_file.exists():
            self.progress_file.unlink()
        self.processed_files.clear()
        self.stats = ProcessingStats()


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None,
                  console: bool = True) -> None:
    """
    Configure logging for the chunking pipeline.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        console: Whether to log to console
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = []

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers,
        force=True
    )

    # Reduce noise from some libraries
    logging.getLogger("unstructured").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def compute_content_hash(content: str) -> str:
    """
    Compute SHA-256 hash of content for deduplication.

    Args:
        content: Text content to hash

    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in text.
    Rough approximation: 1 token ≈ 4 characters for English text.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 4


def create_progress_bar(iterable, desc: str = "Processing",
                       show: bool = True, **kwargs) -> tqdm:
    """
    Create a progress bar with consistent styling.

    Args:
        iterable: Iterable to wrap
        desc: Description for the progress bar
        show: Whether to show the progress bar
        **kwargs: Additional arguments for tqdm

    Returns:
        tqdm progress bar
    """
    if not show:
        return iterable

    # Set default unit only if not provided in kwargs
    if 'unit' not in kwargs:
        kwargs['unit'] = 'item'

    return tqdm(
        iterable,
        desc=desc,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        **kwargs
    )


def validate_file_size(file_path: Path, max_size_mb: int = 50) -> bool:
    """
    Validate that a file is not too large.

    Args:
        file_path: Path to file
        max_size_mb: Maximum file size in megabytes

    Returns:
        True if file size is acceptable, False otherwise
    """
    try:
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            logging.warning(f"File {file_path} is too large ({file_size_mb:.1f}MB > {max_size_mb}MB)")
            return False
        return True
    except Exception as e:
        logging.error(f"Error checking file size for {file_path}: {e}")
        return False


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    import yaml

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading config from {config_path}: {e}")
        raise


def merge_config(base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge configuration dictionaries.

    Args:
        base_config: Base configuration
        overrides: Override values

    Returns:
        Merged configuration
    """
    result = base_config.copy()

    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value

    return result
