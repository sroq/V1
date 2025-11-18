"""
Correctness Evaluator - LLM as a Judge for Correctness.

This module evaluates whether a generated response is factually correct
by comparing it with the ground truth answer using an LLM judge.

Adapted from /04-41-peldakok/02-code/single_turn_evaluation.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from openai import OpenAI
import logging
import re

from config import OPENAI_API_KEY, JUDGE_MODEL, TEMPERATURE_JUDGE, MAX_TOKENS_JUDGE
from .prompts import format_correctness_prompt

# Import cost tracking
sys.path.insert(0, str(Path(__file__).parent.parent))
from cost_metrics import get_cost_tracker

logger = logging.getLogger(__name__)


class CorrectnessEvaluator:
    """
    LLM-based correctness evaluator.

    Uses GPT-4o mini as a judge to evaluate whether a generated response
    is factually correct by comparing it with the ground truth answer.

    Attributes:
        client: OpenAI API client
        model: Judge model name
        temperature: Temperature for judge (0.0 for deterministic)
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the correctness evaluator.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY from config)
        """
        self.client = OpenAI(api_key=api_key or OPENAI_API_KEY)
        self.model = JUDGE_MODEL
        self.temperature = TEMPERATURE_JUDGE
        self.max_tokens = MAX_TOKENS_JUDGE

        # Cost tracking
        self.cost_tracker = get_cost_tracker()

        logger.info(f"CorrectnessEvaluator initialized: {self.model}, temp={self.temperature}")

    def evaluate(
        self,
        ground_truth: str,
        generated_response: str
    ) -> Dict[str, Any]:
        """
        Evaluate correctness of generated response against ground truth.

        Args:
            ground_truth: The reference answer (correct answer)
            generated_response: The AI-generated answer to evaluate

        Returns:
            Dictionary with evaluation results:
                - is_correct: Boolean (True if correct, False if incorrect)
                - decision: String ("CORRECT" or "INCORRECT")
                - reasoning: String (judge's explanation)
                - raw_response: String (full LLM response)
                - error: String (if evaluation failed)

        Example:
            >>> evaluator = CorrectnessEvaluator()
            >>> result = evaluator.evaluate(
            ...     ground_truth="Rudyard Kipling wrote The Jungle Book.",
            ...     generated_response="The author is Rudyard Kipling."
            ... )
            >>> print(result['is_correct'])
            True
            >>> print(result['reasoning'])
        """
        try:
            # Format prompt
            prompt = format_correctness_prompt(ground_truth, generated_response)

            # Call judge LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Track cost
            if response.usage:
                self.cost_tracker.record_llm_cost(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    attributes={"operation": "correctness_evaluation"}
                )

            # Extract response
            raw_response = response.choices[0].message.content.strip()

            # Parse structured response
            parsed = self._parse_judge_response(raw_response)

            logger.debug(f"Correctness evaluation: {parsed['decision']}")

            return {
                'is_correct': parsed['is_correct'],
                'decision': parsed['decision'],
                'reasoning': parsed['reasoning'],
                'raw_response': raw_response,
                'error': None
            }

        except Exception as e:
            logger.error(f"Correctness evaluation failed: {e}")
            return {
                'is_correct': False,
                'decision': "ERROR",
                'reasoning': f"Evaluation failed: {str(e)}",
                'raw_response': "",
                'error': str(e)
            }

    def _parse_judge_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Parse the judge's structured response.

        Expected format:
            REASONING:
            <reasoning text>

            DECISION: CORRECT or INCORRECT

        Args:
            raw_response: Full LLM response text

        Returns:
            Dictionary with parsed fields:
                - decision: "CORRECT" or "INCORRECT"
                - is_correct: Boolean
                - reasoning: Explanation text
        """
        decision = None
        reasoning = None

        # Split by lines
        lines = raw_response.split('\n')

        # Extract reasoning and decision
        current_section = None
        reasoning_lines = []

        for line in lines:
            line_stripped = line.strip()

            if line_stripped.startswith("REASONING:"):
                current_section = "reasoning"
                # Extract reasoning from same line if present
                reasoning_text = line.replace("REASONING:", "").strip()
                if reasoning_text:
                    reasoning_lines.append(reasoning_text)

            elif line_stripped.startswith("DECISION:"):
                current_section = "decision"
                # Extract decision
                decision_text = line.replace("DECISION:", "").strip()
                if "CORRECT" in decision_text.upper() and "INCORRECT" not in decision_text.upper():
                    decision = "CORRECT"
                elif "INCORRECT" in decision_text.upper():
                    decision = "INCORRECT"

            elif current_section == "reasoning" and line_stripped:
                # Continue collecting reasoning lines
                reasoning_lines.append(line_stripped)

        # Join reasoning lines
        if reasoning_lines:
            reasoning = "\n".join(reasoning_lines)
        else:
            reasoning = raw_response  # Fallback: use full response

        # Fallback decision extraction (if not found via DECISION: line)
        if decision is None:
            upper_response = raw_response.upper()
            if "CORRECT" in upper_response and "INCORRECT" not in upper_response:
                decision = "CORRECT"
            elif "INCORRECT" in upper_response:
                decision = "INCORRECT"
            else:
                # Default to INCORRECT if unclear
                logger.warning("Could not parse decision, defaulting to INCORRECT")
                decision = "INCORRECT"

        # Convert to boolean
        is_correct = (decision == "CORRECT")

        return {
            'decision': decision,
            'is_correct': is_correct,
            'reasoning': reasoning if reasoning else raw_response
        }


# ============================================================================
# BATCH EVALUATION UTILITY
# ============================================================================

def evaluate_correctness_batch(
    evaluations: list,
    evaluator: Optional[CorrectnessEvaluator] = None,
    show_progress: bool = True
) -> list:
    """
    Evaluate correctness for a batch of (ground_truth, generated_response) pairs.

    Args:
        evaluations: List of dicts with 'ground_truth' and 'generated_response' keys
        evaluator: CorrectnessEvaluator instance (creates new if None)
        show_progress: Show tqdm progress bar

    Returns:
        List of evaluation result dictionaries

    Example:
        >>> evaluations = [
        ...     {'ground_truth': 'Answer A', 'generated_response': 'Answer A'},
        ...     {'ground_truth': 'Answer B', 'generated_response': 'Wrong answer'}
        ... ]
        >>> results = evaluate_correctness_batch(evaluations)
        >>> print(f"Correctness rate: {sum(r['is_correct'] for r in results) / len(results)}")
    """
    if evaluator is None:
        evaluator = CorrectnessEvaluator()

    results = []

    # Optional progress bar
    iterator = evaluations
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(evaluations, desc="Evaluating correctness")
        except ImportError:
            logger.warning("tqdm not installed, progress bar disabled")

    for item in iterator:
        try:
            result = evaluator.evaluate(
                ground_truth=item['ground_truth'],
                generated_response=item['generated_response']
            )
            # Add original data to result
            result['ground_truth'] = item['ground_truth']
            result['generated_response'] = item['generated_response']
            results.append(result)

        except Exception as e:
            logger.error(f"Batch evaluation item failed: {e}")
            results.append({
                'ground_truth': item.get('ground_truth'),
                'generated_response': item.get('generated_response'),
                'is_correct': False,
                'decision': "ERROR",
                'reasoning': str(e),
                'raw_response': "",
                'error': str(e)
            })

    return results


if __name__ == '__main__':
    # Test the correctness evaluator
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test cases
    test_cases = [
        {
            'name': 'Exact match (CORRECT)',
            'ground_truth': 'Rudyard Kipling wrote The Jungle Book in 1894.',
            'generated_response': 'Rudyard Kipling wrote The Jungle Book in 1894.'
        },
        {
            'name': 'Paraphrase (should be CORRECT)',
            'ground_truth': 'Bagheera is a black panther.',
            'generated_response': 'Bagheera is a panther with black fur.'
        },
        {
            'name': 'Contradiction (INCORRECT)',
            'ground_truth': 'Mowgli was raised by wolves.',
            'generated_response': 'Mowgli was raised by bears.'
        },
        {
            'name': 'Missing key info (INCORRECT)',
            'ground_truth': 'Bagheera paid one bull to buy Mowgli\'s acceptance.',
            'generated_response': 'Bagheera helped Mowgli join the pack.'
        }
    ]

    try:
        logger.info("=" * 80)
        logger.info("Testing CorrectnessEvaluator")
        logger.info("=" * 80)

        evaluator = CorrectnessEvaluator()

        for i, test in enumerate(test_cases, 1):
            logger.info(f"\nTest {i}: {test['name']}")
            logger.info(f"Ground truth: {test['ground_truth']}")
            logger.info(f"Generated: {test['generated_response']}")

            result = evaluator.evaluate(
                ground_truth=test['ground_truth'],
                generated_response=test['generated_response']
            )

            logger.info(f"Decision: {result['decision']}")
            logger.info(f"Reasoning: {result['reasoning'][:200]}...")

        logger.info("\n" + "=" * 80)
        logger.info("✓ CorrectnessEvaluator test complete")

    except Exception as e:
        logger.error(f"✗ CorrectnessEvaluator test failed: {e}")
        sys.exit(1)
