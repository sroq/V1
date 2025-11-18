#!/usr/bin/env python3
"""
Evaluate Relevance - LLM as a Judge.

This script evaluates the relevance of assistant responses to the user's question
using an LLM judge.

Process:
1. Load assistant_responses.json
2. For each response:
   - Evaluate if response is relevant to the question
   - Get LLM judge decision (RELEVANT/IRRELEVANT + reasoning)
3. Save relevance_evaluations.json

Usage:
    python scripts/4_evaluate_relevance.py
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_judge import RelevanceEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_assistant_responses(path: Path) -> Dict[str, Any]:
    """
    Load assistant responses from JSON file.

    Args:
        path: Path to assistant_responses.json

    Returns:
        Assistant responses dictionary

    Raises:
        FileNotFoundError: If responses file not found
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Assistant responses not found: {path}\n"
            "Please run scripts/2_run_assistant.py first"
        )

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Loaded assistant responses: {data['metadata']['total_questions']} entries")
    return data


def evaluate_relevance(
    responses: List[Dict[str, Any]],
    evaluator: RelevanceEvaluator
) -> List[Dict[str, Any]]:
    """
    Evaluate relevance for all responses.

    Args:
        responses: List of response dictionaries
        evaluator: RelevanceEvaluator instance

    Returns:
        List of evaluation dictionaries
    """
    evaluations = []

    logger.info(f"\nEvaluating relevance for {len(responses)} responses...")

    for i, response in enumerate(responses, 1):
        entry_id = response['id']
        question = response['question']
        generated = response.get('generated_response')

        logger.info(f"\n[{i}/{len(responses)}] {entry_id}: {question}")

        # Skip if no generated response (error case)
        if not generated:
            logger.warning(f"  ⚠ Skipping: No generated response")
            evaluations.append({
                'id': entry_id,
                'question': question,
                'generated_response': generated,
                'is_relevant': False,
                'decision': 'SKIPPED',
                'reasoning': 'No generated response (assistant failed)',
                'category': response.get('category'),
                'difficulty': response.get('difficulty'),
                'error': response.get('error'),
                'timestamp': datetime.now().isoformat()
            })
            continue

        try:
            # Evaluate relevance
            result = evaluator.evaluate(
                query=question,
                generated_response=generated
            )

            # Create evaluation entry
            evaluation = {
                'id': entry_id,
                'question': question,
                'generated_response': generated,
                'is_relevant': result['is_relevant'],
                'decision': result['decision'],
                'reasoning': result['reasoning'],
                'raw_response': result['raw_response'],
                'category': response.get('category'),
                'difficulty': response.get('difficulty'),
                'timestamp': datetime.now().isoformat()
            }

            evaluations.append(evaluation)

            # Preview
            logger.info(f"  Decision: {result['decision']}")
            logger.info(f"  Reasoning: {result['reasoning'][:150]}...")

        except Exception as e:
            logger.error(f"  ✗ Evaluation failed: {e}")
            evaluations.append({
                'id': entry_id,
                'question': question,
                'generated_response': generated,
                'is_relevant': False,
                'decision': 'ERROR',
                'reasoning': f'Evaluation failed: {str(e)}',
                'category': response.get('category'),
                'difficulty': response.get('difficulty'),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

    return evaluations


def main():
    """
    Main script entry point.
    """
    logger.info("=" * 80)
    logger.info("RELEVANCE EVALUATION - LLM AS A JUDGE")
    logger.info("=" * 80)

    # Paths
    data_dir = Path(__file__).parent.parent / 'data'
    responses_path = data_dir / 'assistant_responses.json'
    output_path = data_dir / 'relevance_evaluations.json'

    # Load assistant responses
    try:
        responses_data = load_assistant_responses(responses_path)
    except FileNotFoundError as e:
        logger.error(f"✗ {e}")
        sys.exit(1)

    responses = responses_data['responses']

    # Initialize evaluator
    logger.info("\nInitializing relevance evaluator (LLM judge)...")
    evaluator = RelevanceEvaluator()

    # Evaluate all responses
    evaluations = evaluate_relevance(responses, evaluator)

    # Calculate metrics
    total = len(evaluations)
    relevant = sum(1 for e in evaluations if e['is_relevant'])
    relevance_rate = relevant / total * 100 if total > 0 else 0

    logger.info(f"\n✓ Relevance evaluation complete")
    logger.info(f"  Relevant: {relevant}/{total} ({relevance_rate:.1f}%)")

    # Save evaluations
    output_data = {
        'metadata': {
            'dataset_id': responses_data['metadata']['dataset_id'],
            'total_evaluations': total,
            'relevant_count': relevant,
            'irrelevant_count': total - relevant,
            'relevance_rate': relevance_rate,
            'evaluated_at': datetime.now().isoformat()
        },
        'evaluations': evaluations
    }

    data_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✓ Evaluations saved to: {output_path}")

    # Summary by category
    logger.info("\nRelevance by category:")
    category_stats = {}
    for e in evaluations:
        cat = e.get('category', 'unknown')
        if cat not in category_stats:
            category_stats[cat] = {'total': 0, 'relevant': 0}
        category_stats[cat]['total'] += 1
        if e['is_relevant']:
            category_stats[cat]['relevant'] += 1

    for cat, stats in sorted(category_stats.items()):
        rate = stats['relevant'] / stats['total'] * 100 if stats['total'] > 0 else 0
        logger.info(f"  {cat}: {stats['relevant']}/{stats['total']} ({rate:.1f}%)")

    # Summary by difficulty
    logger.info("\nRelevance by difficulty:")
    difficulty_stats = {}
    for e in evaluations:
        diff = e.get('difficulty', 'unknown')
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {'total': 0, 'relevant': 0}
        difficulty_stats[diff]['total'] += 1
        if e['is_relevant']:
            difficulty_stats[diff]['relevant'] += 1

    for diff, stats in sorted(difficulty_stats.items()):
        rate = stats['relevant'] / stats['total'] * 100 if stats['total'] > 0 else 0
        logger.info(f"  {diff}: {stats['relevant']}/{stats['total']} ({rate:.1f}%)")

    logger.info("\n" + "=" * 80)
    logger.info("✓ RELEVANCE EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info("\nNext step: Analyze results and generate report")
    logger.info("  python scripts/5_analyze_results.py")


if __name__ == '__main__':
    main()
