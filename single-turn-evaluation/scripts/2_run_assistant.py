#!/usr/bin/env python3
"""
Run Assistant on Golden Dataset.

This script runs the RAG assistant on all questions in the golden dataset
and saves the generated responses with metadata.

Process:
1. Load golden_dataset.json
2. For each question:
   - Run full RAG pipeline (embedding → search → response)
   - Capture response and metadata
3. Save assistant_responses.json with all results

Usage:
    python scripts/2_run_assistant.py
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TOP_K_CHUNKS, SIMILARITY_THRESHOLD
from llm_judge import AssistantRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_golden_dataset(path: Path) -> Dict[str, Any]:
    """
    Load golden dataset from JSON file.

    Args:
        path: Path to golden_dataset.json

    Returns:
        Golden dataset dictionary

    Raises:
        FileNotFoundError: If dataset file not found
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Golden dataset not found: {path}\n"
            "Please run scripts/1_generate_golden_dataset.py first"
        )

    with open(path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    logger.info(f"Loaded golden dataset: {dataset['metadata']['total_pairs']} entries")
    return dataset


def run_assistant_on_dataset(
    dataset: Dict[str, Any],
    runner: AssistantRunner
) -> List[Dict[str, Any]]:
    """
    Run assistant on all questions in the golden dataset.

    Args:
        dataset: Golden dataset dictionary
        runner: AssistantRunner instance

    Returns:
        List of response dictionaries
    """
    entries = dataset['entries']
    responses = []

    logger.info(f"\nRunning assistant on {len(entries)} questions...")
    logger.info(f"Parameters: top_k={TOP_K_CHUNKS}, threshold={SIMILARITY_THRESHOLD}")

    for i, entry in enumerate(entries, 1):
        entry_id = entry['id']
        question = entry['question']

        logger.info(f"\n[{i}/{len(entries)}] {entry_id}: {question}")

        try:
            # Run RAG pipeline
            result = runner.run_full_pipeline(
                query=question,
                top_k=TOP_K_CHUNKS,
                similarity_threshold=SIMILARITY_THRESHOLD
            )

            # Create response entry
            response_entry = {
                'id': entry_id,
                'question': question,
                'generated_response': result['generated_response'],
                'ground_truth_answer': entry['ground_truth_answer'],
                'category': entry['category'],
                'difficulty': entry['difficulty'],
                'source_chunk_id': entry['source_chunk_id'],
                'retrieved_chunks': [
                    {
                        'chunk_id': str(chunk['id']),  # Convert UUID to string
                        'similarity_score': chunk['similarity_score'],
                        'content_preview': chunk['content'][:200] + '...'
                    }
                    for chunk in result['retrieved_chunks']
                ],
                'pipeline_metadata': result['pipeline_metadata'],
                'response_metadata': result['response_metadata'],
                'timestamp': datetime.now().isoformat()
            }

            responses.append(response_entry)

            # Preview
            logger.info(f"  Generated: {result['generated_response'][:150]}...")
            logger.info(f"  Retrieved: {len(result['retrieved_chunks'])} chunks")
            logger.info(f"  Latency: {result['pipeline_metadata']['total_latency_ms']}ms")

        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            # Add error entry
            responses.append({
                'id': entry_id,
                'question': question,
                'generated_response': None,
                'ground_truth_answer': entry['ground_truth_answer'],
                'category': entry['category'],
                'difficulty': entry['difficulty'],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

    return responses


def main():
    """
    Main script entry point.
    """
    logger.info("=" * 80)
    logger.info("RUN ASSISTANT ON GOLDEN DATASET")
    logger.info("=" * 80)

    # Paths
    data_dir = Path(__file__).parent.parent / 'data'
    golden_path = data_dir / 'golden_dataset.json'
    output_path = data_dir / 'assistant_responses.json'

    # Load golden dataset
    try:
        dataset = load_golden_dataset(golden_path)
    except FileNotFoundError as e:
        logger.error(f"✗ {e}")
        sys.exit(1)

    # Initialize assistant runner
    logger.info("\nInitializing RAG assistant...")
    runner = AssistantRunner()

    # Run assistant on all questions
    responses = run_assistant_on_dataset(dataset, runner)

    # Calculate success rate
    successful = sum(1 for r in responses if r.get('generated_response') is not None)
    success_rate = successful / len(responses) * 100 if responses else 0

    logger.info(f"\n✓ Completed: {successful}/{len(responses)} successful ({success_rate:.1f}%)")

    # Save responses
    output_data = {
        'metadata': {
            'dataset_id': dataset['metadata']['dataset_id'],
            'total_questions': len(responses),
            'successful_responses': successful,
            'success_rate': success_rate,
            'rag_parameters': {
                'top_k': TOP_K_CHUNKS,
                'similarity_threshold': SIMILARITY_THRESHOLD
            },
            'generated_at': datetime.now().isoformat()
        },
        'responses': responses
    }

    data_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✓ Responses saved to: {output_path}")

    # Summary statistics
    logger.info("\nSummary by category:")
    category_stats = {}
    for r in responses:
        cat = r.get('category', 'unknown')
        if cat not in category_stats:
            category_stats[cat] = {'total': 0, 'successful': 0}
        category_stats[cat]['total'] += 1
        if r.get('generated_response'):
            category_stats[cat]['successful'] += 1

    for cat, stats in category_stats.items():
        rate = stats['successful'] / stats['total'] * 100 if stats['total'] > 0 else 0
        logger.info(f"  {cat}: {stats['successful']}/{stats['total']} ({rate:.1f}%)")

    # Average latency
    latencies = [
        r['pipeline_metadata']['total_latency_ms']
        for r in responses
        if 'pipeline_metadata' in r and 'total_latency_ms' in r['pipeline_metadata']
    ]
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        logger.info(f"\nAverage latency: {avg_latency:.0f}ms")

    logger.info("\n" + "=" * 80)
    logger.info("✓ ASSISTANT EXECUTION COMPLETE")
    logger.info("=" * 80)
    logger.info("\nNext step: Run correctness evaluation")
    logger.info("  python scripts/3_evaluate_correctness.py")


if __name__ == '__main__':
    main()
