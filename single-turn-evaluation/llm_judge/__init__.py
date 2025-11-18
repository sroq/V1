"""
LLM Judge Értékelési Csomag.

Ez a csomag LLM-alapú értékelési eszközöket biztosít a RAG asszisztens válaszokhoz:
- CorrectnessEvaluator: Ténybeli helyesség értékelése a ground truth ellen
- RelevanceEvaluator: Relevancia értékelése a felhasználó kérdéséhez
- AssistantRunner: A teljes RAG pipeline futtatása
- Adatbázis segédeszközök: PostgreSQL + pgvector kapcsolat és keresés
"""

from .correctness_evaluator import CorrectnessEvaluator, evaluate_correctness_batch
from .relevance_evaluator import RelevanceEvaluator, evaluate_relevance_batch
from .assistant_runner import AssistantRunner, run_assistant_batch
from .database import (
    get_db_connection,
    fetch_chunk_by_id,
    search_similar_chunks,
    fetch_all_chunks,
    get_database_stats
)
from .prompts import (
    format_correctness_prompt,
    format_relevance_prompt,
    format_ground_truth_prompt,
    build_assistant_user_prompt
)

__all__ = [
    # Értékelők
    'CorrectnessEvaluator',
    'RelevanceEvaluator',
    'AssistantRunner',

    # Batch segédeszközök
    'evaluate_correctness_batch',
    'evaluate_relevance_batch',
    'run_assistant_batch',

    # Adatbázis
    'get_db_connection',
    'fetch_chunk_by_id',
    'search_similar_chunks',
    'fetch_all_chunks',
    'get_database_stats',

    # Promptok
    'format_correctness_prompt',
    'format_relevance_prompt',
    'format_ground_truth_prompt',
    'build_assistant_user_prompt',
]
