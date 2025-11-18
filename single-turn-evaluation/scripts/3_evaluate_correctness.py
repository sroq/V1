#!/usr/bin/env python3
"""
Evaluate Correctness - LLM as a Judge.

This script evaluates the correctness of assistant responses by comparing them
with ground truth answers using an LLM judge.

Process:
1. Load assistant_responses.json
2. For each response:
   - Compare generated response with ground truth
   - Get LLM judge decision (CORRECT/INCORRECT + reasoning)
3. Save correctness_evaluations.json

Usage:
    python scripts/3_evaluate_correctness.py
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_judge import CorrectnessEvaluator

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


def evaluate_correctness(
    responses: List[Dict[str, Any]],
    evaluator: CorrectnessEvaluator
) -> List[Dict[str, Any]]:
    """
    Evaluate correctness for all responses.

    Args:
        responses: List of response dictionaries
        evaluator: CorrectnessEvaluator instance

    Returns:
        List of evaluation dictionaries
    """
    evaluations = []

    logger.info(f"\nEvaluating correctness for {len(responses)} responses...")

    for i, response in enumerate(responses, 1):
        entry_id = response['id']
        question = response['question']
        generated = response.get('generated_response')
        ground_truth = response.get('ground_truth_answer')

        logger.info(f"\n[{i}/{len(responses)}] {entry_id}: {question}")

        # Skip if no generated response (error case)
        if not generated:
            logger.warning(f"  ⚠ Skipping: No generated response")
            evaluations.append({
                'id': entry_id,
                'question': question,
                'generated_response': generated,
                'ground_truth_answer': ground_truth,
                'is_correct': False,
                'decision': 'SKIPPED',
                'reasoning': 'No generated response (assistant failed)',
                'category': response.get('category'),
                'difficulty': response.get('difficulty'),
                'error': response.get('error'),
                'timestamp': datetime.now().isoformat()
            })
            continue

        try:
            # Evaluate correctness
            result = evaluator.evaluate(
                ground_truth=ground_truth,
                generated_response=generated
            )

            # Create evaluation entry
            evaluation = {
                'id': entry_id,
                'question': question,
                'generated_response': generated,
                'ground_truth_answer': ground_truth,
                'is_correct': result['is_correct'],
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
                'ground_truth_answer': ground_truth,
                'is_correct': False,
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
    logger.info("CORRECTNESS EVALUATION - LLM AS A JUDGE")
    logger.info("=" * 80)

    # Paths
    data_dir = Path(__file__).parent.parent / 'data'
    responses_path = data_dir / 'assistant_responses.json'
    output_path = data_dir / 'correctness_evaluations.json'

    # Load assistant responses
    try:
        responses_data = load_assistant_responses(responses_path)
    except FileNotFoundError as e:
        logger.error(f"✗ {e}")
        sys.exit(1)

    responses = responses_data['responses']

    # Initialize evaluator
    logger.info("\nInitializing correctness evaluator (LLM judge)...")
    evaluator = CorrectnessEvaluator()

    # Evaluate all responses
    evaluations = evaluate_correctness(responses, evaluator)

    # Calculate metrics
    total = len(evaluations)
    correct = sum(1 for e in evaluations if e['is_correct'])
    correctness_rate = correct / total * 100 if total > 0 else 0

    logger.info(f"\n✓ Correctness evaluation complete")
    logger.info(f"  Correct: {correct}/{total} ({correctness_rate:.1f}%)")

    # Save evaluations
    output_data = {
        'metadata': {
            'dataset_id': responses_data['metadata']['dataset_id'],
            'total_evaluations': total,
            'correct_count': correct,
            'incorrect_count': total - correct,
            'correctness_rate': correctness_rate,
            'evaluated_at': datetime.now().isoformat()
        },
        'evaluations': evaluations
    }

    data_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✓ Evaluations saved to: {output_path}")

    # Summary by category
    logger.info("\nCorrectness by category:")
    category_stats = {}
    for e in evaluations:
        cat = e.get('category', 'unknown')
        if cat not in category_stats:
            category_stats[cat] = {'total': 0, 'correct': 0}
        category_stats[cat]['total'] += 1
        if e['is_correct']:
            category_stats[cat]['correct'] += 1

    for cat, stats in sorted(category_stats.items()):
        rate = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        logger.info(f"  {cat}: {stats['correct']}/{stats['total']} ({rate:.1f}%)")

    # Summary by difficulty
    logger.info("\nCorrectness by difficulty:")
    difficulty_stats = {}
    for e in evaluations:
        diff = e.get('difficulty', 'unknown')
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {'total': 0, 'correct': 0}
        difficulty_stats[diff]['total'] += 1
        if e['is_correct']:
            difficulty_stats[diff]['correct'] += 1

    for diff, stats in sorted(difficulty_stats.items()):
        rate = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        logger.info(f"  {diff}: {stats['correct']}/{stats['total']} ({rate:.1f}%)")

    logger.info("\n" + "=" * 80)
    logger.info("✓ CORRECTNESS EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info("\nNext step: Run relevance evaluation")
    logger.info("  python scripts/4_evaluate_relevance.py")


if __name__ == '__main__':
    main()
