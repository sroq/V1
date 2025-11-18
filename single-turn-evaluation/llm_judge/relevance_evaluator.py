"""
Relevance Evaluator - LLM as a Judge for Relevance.

This module evaluates whether a generated response is relevant to the user's
question using an LLM judge.

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
from .prompts import format_relevance_prompt

# Import cost tracking
sys.path.insert(0, str(Path(__file__).parent.parent))
from cost_metrics import get_cost_tracker

logger = logging.getLogger(__name__)


class RelevanceEvaluator:
    """
    LLM-based relevance evaluator.

    Uses GPT-4o mini as a judge to evaluate whether a generated response
    is relevant to the user's question.

    Attributes:
        client: OpenAI API client
        model: Judge model name
        temperature: Temperature for judge (0.0 for deterministic)
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the relevance evaluator.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY from config)
        """
        self.client = OpenAI(api_key=api_key or OPENAI_API_KEY)
        self.model = JUDGE_MODEL
        self.temperature = TEMPERATURE_JUDGE
        self.max_tokens = MAX_TOKENS_JUDGE

        # Cost tracking
        self.cost_tracker = get_cost_tracker()

        logger.info(f"RelevanceEvaluator initialized: {self.model}, temp={self.temperature}")

    def evaluate(
        self,
        query: str,
        generated_response: str
    ) -> Dict[str, Any]:
        """
        Evaluate relevance of generated response to the user's question.

        Args:
            query: The user's question
            generated_response: The AI-generated answer to evaluate

        Returns:
            Dictionary with evaluation results:
                - is_relevant: Boolean (True if relevant, False if irrelevant)
                - decision: String ("RELEVANT" or "IRRELEVANT")
                - reasoning: String (judge's explanation)
                - raw_response: String (full LLM response)
                - error: String (if evaluation failed)

        Example:
            >>> evaluator = RelevanceEvaluator()
            >>> result = evaluator.evaluate(
            ...     query="Who is Mowgli?",
            ...     generated_response="Mowgli is a human child raised by wolves."
            ... )
            >>> print(result['is_relevant'])
            True
            >>> print(result['reasoning'])
        """
        try:
            # Format prompt
            prompt = format_relevance_prompt(query, generated_response)

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
                    attributes={"operation": "relevance_evaluation"}
                )

            # Extract response
            raw_response = response.choices[0].message.content.strip()

            # Parse structured response
            parsed = self._parse_judge_response(raw_response)

            logger.debug(f"Relevance evaluation: {parsed['decision']}")

            return {
                'is_relevant': parsed['is_relevant'],
                'decision': parsed['decision'],
                'reasoning': parsed['reasoning'],
                'raw_response': raw_response,
                'error': None
            }

        except Exception as e:
            logger.error(f"Relevance evaluation failed: {e}")
            return {
                'is_relevant': False,
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

            DECISION: RELEVANT or IRRELEVANT

        Args:
            raw_response: Full LLM response text

        Returns:
            Dictionary with parsed fields:
                - decision: "RELEVANT" or "IRRELEVANT"
                - is_relevant: Boolean
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
                if "RELEVANT" in decision_text.upper() and "IRRELEVANT" not in decision_text.upper():
                    decision = "RELEVANT"
                elif "IRRELEVANT" in decision_text.upper():
                    decision = "IRRELEVANT"

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
            if "RELEVANT" in upper_response and "IRRELEVANT" not in upper_response:
                decision = "RELEVANT"
            elif "IRRELEVANT" in upper_response:
                decision = "IRRELEVANT"
            else:
                # Default to IRRELEVANT if unclear
                logger.warning("Could not parse decision, defaulting to IRRELEVANT")
                decision = "IRRELEVANT"

        # Convert to boolean
        is_relevant = (decision == "RELEVANT")

        return {
            'decision': decision,
            'is_relevant': is_relevant,
            'reasoning': reasoning if reasoning else raw_response
        }


# ============================================================================
# BATCH EVALUATION UTILITY
# ============================================================================

def evaluate_relevance_batch(
    evaluations: list,
    evaluator: Optional[RelevanceEvaluator] = None,
    show_progress: bool = True
) -> list:
    """
    Evaluate relevance for a batch of (query, generated_response) pairs.

    Args:
        evaluations: List of dicts with 'query' and 'generated_response' keys
        evaluator: RelevanceEvaluator instance (creates new if None)
        show_progress: Show tqdm progress bar

    Returns:
        List of evaluation result dictionaries

    Example:
        >>> evaluations = [
        ...     {'query': 'Who is Mowgli?', 'generated_response': 'Mowgli is a human child.'},
        ...     {'query': 'Who is Bagheera?', 'generated_response': 'Rudyard Kipling is the author.'}
        ... ]
        >>> results = evaluate_relevance_batch(evaluations)
        >>> print(f"Relevance rate: {sum(r['is_relevant'] for r in results) / len(results)}")
    """
    if evaluator is None:
        evaluator = RelevanceEvaluator()

    results = []

    # Optional progress bar
    iterator = evaluations
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(evaluations, desc="Evaluating relevance")
        except ImportError:
            logger.warning("tqdm not installed, progress bar disabled")

    for item in iterator:
        try:
            result = evaluator.evaluate(
                query=item['query'],
                generated_response=item['generated_response']
            )
            # Add original data to result
            result['query'] = item['query']
            result['generated_response'] = item['generated_response']
            results.append(result)

        except Exception as e:
            logger.error(f"Batch evaluation item failed: {e}")
            results.append({
                'query': item.get('query'),
                'generated_response': item.get('generated_response'),
                'is_relevant': False,
                'decision': "ERROR",
                'reasoning': str(e),
                'raw_response': "",
                'error': str(e)
            })

    return results


if __name__ == '__main__':
    # Test the relevance evaluator
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test cases
    test_cases = [
        {
            'name': 'Direct answer (RELEVANT)',
            'query': 'Who is Mowgli?',
            'generated_response': 'Mowgli is a human child who was raised by wolves in the jungle.'
        },
        {
            'name': 'Off-topic answer (IRRELEVANT)',
            'query': 'Who is Mowgli?',
            'generated_response': 'The Jungle Book was written by Rudyard Kipling in 1894.'
        },
        {
            'name': 'Partial but relevant (RELEVANT)',
            'query': 'Why did Bagheera pay a bull for Mowgli?',
            'generated_response': 'Bagheera wanted to help Mowgli join the wolf pack.'
        },
        {
            'name': 'Generic tangent (IRRELEVANT)',
            'query': 'How did Mowgli escape from the monkeys?',
            'generated_response': 'Monkeys are interesting animals that live in trees and eat fruit.'
        }
    ]

    try:
        logger.info("=" * 80)
        logger.info("Testing RelevanceEvaluator")
        logger.info("=" * 80)

        evaluator = RelevanceEvaluator()

        for i, test in enumerate(test_cases, 1):
            logger.info(f"\nTest {i}: {test['name']}")
            logger.info(f"Query: {test['query']}")
            logger.info(f"Generated: {test['generated_response']}")

            result = evaluator.evaluate(
                query=test['query'],
                generated_response=test['generated_response']
            )

            logger.info(f"Decision: {result['decision']}")
            logger.info(f"Reasoning: {result['reasoning'][:200]}...")

        logger.info("\n" + "=" * 80)
        logger.info("✓ RelevanceEvaluator test complete")

    except Exception as e:
        logger.error(f"✗ RelevanceEvaluator test failed: {e}")
        sys.exit(1)
