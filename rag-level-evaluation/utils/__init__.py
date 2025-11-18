"""
Utility modules for RAG evaluation system.
"""
from .db_utils import get_db_connection, fetch_all_chunks, search_similar_chunks
from .openai_utils import generate_embedding, generate_question
from .metrics import (
    calculate_metrics,
    aggregate_metrics,
    format_results,
    results_to_dataframe,
    calculate_metrics_by_strategy,
    get_similarity_distributions,
    calculate_chunk_size_consistency,
    calculate_retrieval_by_chunk_size,
    calculate_position_stability
)

__all__ = [
    'get_db_connection',
    'fetch_all_chunks',
    'search_similar_chunks',
    'generate_embedding',
    'generate_question',
    'calculate_metrics',
    'aggregate_metrics',
    'format_results',
    'results_to_dataframe',
    'calculate_metrics_by_strategy',
    'get_similarity_distributions',
    'calculate_chunk_size_consistency',
    'calculate_retrieval_by_chunk_size',
    'calculate_position_stability',
]
