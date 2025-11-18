#!/usr/bin/env python3
"""
Golden Dataset Generation Script.

This script generates a golden dataset of 20-25 Q&A pairs with ground truth answers
for evaluating the RAG assistant.

Process:
1. Load generated_questions.json (from rag-level-evaluation) or manual input
2. Select representative questions across categories
3. Generate ground truth answers using LLM
4. Manual review and refinement
5. Save golden dataset to data/golden_dataset.json

Usage:
    python scripts/1_generate_golden_dataset.py
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    OPENAI_API_KEY,
    ASSISTANT_MODEL,
    TEMPERATURE_ASSISTANT,
    GOLDEN_DATASET_SIZE_TARGET,
    CATEGORY_DISTRIBUTION
)
from llm_judge import (
    get_db_connection,
    fetch_chunk_by_id,
    fetch_all_chunks,
    format_ground_truth_prompt
)
from openai import OpenAI

# Import cost tracking
sys.path.insert(0, str(Path(__file__).parent.parent))
from cost_metrics import get_cost_tracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GoldenDatasetGenerator:
    """
    Golden dataset generator for RAG evaluation.
    """

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = ASSISTANT_MODEL
        self.temperature = TEMPERATURE_ASSISTANT

        # Cost tracking
        self.cost_tracker = get_cost_tracker()

    def load_generated_questions(self, path: str) -> Optional[List[Dict]]:
        """
        Load generated questions from rag-level-evaluation.

        Args:
            path: Path to generated_questions.json

        Returns:
            List of question dictionaries or None if file not found
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Handle two possible formats:
                # 1. {"questions": [...]} - array under 'questions' key
                # 2. {"chunk_id": {...}, ...} - dictionary with chunk IDs as keys

                if 'questions' in data:
                    # Format 1: Array under 'questions' key
                    questions = data['questions']
                elif isinstance(data, dict):
                    # Format 2: Dictionary with chunk IDs as keys
                    # Convert to list and add chunk_id if not present
                    questions = []
                    for chunk_id, item in data.items():
                        if isinstance(item, dict) and 'question' in item:
                            # Ensure chunk_id is in the item
                            if 'chunk_id' not in item:
                                item['chunk_id'] = chunk_id
                            # Use source_chunk_id for consistency
                            item['source_chunk_id'] = item.get('chunk_id', chunk_id)
                            questions.append(item)
                else:
                    questions = []

                logger.info(f"Loaded {len(questions)} questions from {path}")
                return questions
        except FileNotFoundError:
            logger.warning(f"File not found: {path}")
            return None
        except Exception as e:
            logger.error(f"Error loading questions: {e}")
            return None

    def generate_ground_truth_answer(
        self,
        question: str,
        chunk_content: str
    ) -> str:
        """
        Generate ground truth answer using LLM.

        Args:
            question: The question to answer
            chunk_content: The text excerpt to base the answer on

        Returns:
            Ground truth answer (1-3 sentences)
        """
        try:
            prompt = format_ground_truth_prompt(question, chunk_content)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=150
            )

            # Track cost
            if response.usage:
                self.cost_tracker.record_llm_cost(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    attributes={"operation": "golden_dataset_generation"}
                )

            answer = response.choices[0].message.content.strip()
            logger.debug(f"Generated answer: {answer[:100]}...")
            return answer

        except Exception as e:
            logger.error(f"Error generating ground truth: {e}")
            raise

    def categorize_question(self, question: str, chunk_content: str = "") -> str:
        """
        Categorize question based on complexity.

        Categories:
        - factual: Simple factual recall
        - detail: Detailed information extraction
        - comprehension: Understanding and inference
        - multi_chunk: Requires multiple chunks to answer

        Args:
            question: The question text
            chunk_content: Optional chunk content for context

        Returns:
            Category string
        """
        question_lower = question.lower()

        # Simple heuristics for categorization
        if any(word in question_lower for word in ['who is', 'what is', 'when was', 'where is']):
            return 'factual'
        elif any(word in question_lower for word in ['why', 'how did', 'what caused']):
            return 'comprehension'
        elif any(word in question_lower for word in ['describe', 'explain', 'what are']):
            return 'detail'
        else:
            return 'detail'  # Default

    def assess_difficulty(self, question: str, chunk_content: str = "") -> str:
        """
        Assess question difficulty.

        Levels:
        - easy: Direct answer in chunk
        - medium: Requires some inference
        - hard: Complex reasoning or multiple chunks

        Args:
            question: The question text
            chunk_content: Optional chunk content

        Returns:
            Difficulty string
        """
        # Simple heuristics
        question_lower = question.lower()

        if any(word in question_lower for word in ['who', 'what', 'when', 'where']):
            return 'easy'
        elif any(word in question_lower for word in ['why', 'how']):
            return 'medium'
        elif any(word in question_lower for word in ['compare', 'contrast', 'analyze']):
            return 'hard'
        else:
            return 'medium'  # Default

    def create_golden_entry(
        self,
        entry_id: str,
        question: str,
        source_chunk_id: str,
        chunk_content: Optional[str] = None,
        ground_truth_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a golden dataset entry.

        Args:
            entry_id: Unique entry ID (e.g., "q001")
            question: The question text
            source_chunk_id: Source chunk UUID
            chunk_content: Chunk content (if None, fetch from database)
            ground_truth_answer: Ground truth answer (generated if None)

        Returns:
            Golden dataset entry dictionary
        """
        # Use provided chunk_content or fetch from database
        if chunk_content is None:
            # Fetch source chunk from database
            chunk = fetch_chunk_by_id(source_chunk_id)
            if not chunk:
                raise ValueError(f"Chunk not found: {source_chunk_id}")
            chunk_content = chunk['content']

        # Generate ground truth if not provided
        if ground_truth_answer is None:
            logger.info(f"Generating ground truth for: {question}")
            ground_truth_answer = self.generate_ground_truth_answer(
                question, chunk_content
            )

        # Categorize and assess difficulty
        category = self.categorize_question(question, chunk_content)
        difficulty = self.assess_difficulty(question, chunk_content)

        entry = {
            'id': entry_id,
            'question': question,
            'ground_truth_answer': ground_truth_answer,
            'source_chunk_id': source_chunk_id,
            'category': category,
            'difficulty': difficulty,
            'metadata': {
                'chunk_preview': chunk_content[:200] + '...',
                'created_at': datetime.now().isoformat()
            }
        }

        return entry

    def interactive_selection(
        self,
        questions: List[Dict]
    ) -> List[Dict]:
        """
        Interactive question selection.

        Args:
            questions: List of question dictionaries

        Returns:
            Selected questions
        """
        print("\n" + "=" * 80)
        print("INTERACTIVE QUESTION SELECTION")
        print("=" * 80)
        print(f"\nTarget: {GOLDEN_DATASET_SIZE_TARGET} questions")
        print(f"Distribution: {CATEGORY_DISTRIBUTION}")
        print("\nAvailable questions:")

        for i, q in enumerate(questions[:50], 1):  # Show first 50
            print(f"{i:3d}. {q.get('question', 'N/A')}")

        print("\nEnter question indices (comma-separated) or 'auto' for automatic selection:")
        user_input = input("> ").strip()

        if user_input.lower() == 'auto':
            # Automatic selection
            selected = questions[:GOLDEN_DATASET_SIZE_TARGET]
            logger.info(f"Auto-selected {len(selected)} questions")
            return selected
        else:
            # Manual selection
            try:
                indices = [int(x.strip()) - 1 for x in user_input.split(',')]
                selected = [questions[i] for i in indices if 0 <= i < len(questions)]
                logger.info(f"Manually selected {len(selected)} questions")
                return selected
            except Exception as e:
                logger.error(f"Invalid input: {e}")
                return questions[:GOLDEN_DATASET_SIZE_TARGET]


def main():
    """
    Main script entry point.
    """
    logger.info("=" * 80)
    logger.info("GOLDEN DATASET GENERATION")
    logger.info("=" * 80)

    generator = GoldenDatasetGenerator()

    # Try to load existing questions
    questions_path = "../rag-level-evaluation/data/generated_questions.json"
    questions = generator.load_generated_questions(questions_path)

    if questions is None:
        logger.warning("\nNo generated_questions.json found!")
        logger.info("Options:")
        logger.info("1. Run rag-level-evaluation pipeline first")
        logger.info("2. Manually create questions below")
        print("\nProceed with manual question creation? (y/n): ", end='')
        choice = input().strip().lower()

        if choice != 'y':
            logger.info("Exiting. Please run rag-level-evaluation first.")
            sys.exit(0)

        # Manual question creation
        logger.info("\nManual question creation mode")
        logger.info("Fetching sample chunks from database...")

        chunks = fetch_all_chunks(limit=10)
        if not chunks:
            logger.error("No chunks found in database!")
            sys.exit(1)

        # Create sample questions manually
        questions = [
            {
                'question': 'Who is the author of The Jungle Book?',
                'source_chunk_id': chunks[0]['id']
            },
            {
                'question': 'Who is Mowgli?',
                'source_chunk_id': chunks[1]['id'] if len(chunks) > 1 else chunks[0]['id']
            }
        ]
        logger.info(f"Created {len(questions)} sample questions")

    # Interactive selection
    selected_questions = generator.interactive_selection(questions)

    if len(selected_questions) == 0:
        logger.error("No questions selected!")
        sys.exit(1)

    # Generate golden dataset
    logger.info(f"\nGenerating golden dataset with {len(selected_questions)} entries...")

    golden_entries = []

    for i, q in enumerate(selected_questions, 1):
        entry_id = f"q{i:03d}"
        question_text = q.get('question', '')
        source_chunk_id = q.get('source_chunk_id', q.get('chunk_id', ''))
        chunk_content = q.get('chunk_content', None)  # Get chunk content from question data

        try:
            logger.info(f"\n[{i}/{len(selected_questions)}] Processing: {question_text}")

            entry = generator.create_golden_entry(
                entry_id=entry_id,
                question=question_text,
                source_chunk_id=source_chunk_id,
                chunk_content=chunk_content  # Pass chunk content to avoid DB query
            )

            golden_entries.append(entry)

            # Show preview
            print(f"  Question: {entry['question']}")
            print(f"  Ground Truth: {entry['ground_truth_answer']}")
            print(f"  Category: {entry['category']}, Difficulty: {entry['difficulty']}")

        except Exception as e:
            logger.error(f"Failed to process question {i}: {e}")
            continue

    # Create golden dataset
    golden_dataset = {
        'metadata': {
            'dataset_id': 'jungle_book_golden_v1',
            'created_at': datetime.now().isoformat(),
            'total_pairs': len(golden_entries),
            'target_size': GOLDEN_DATASET_SIZE_TARGET,
            'category_distribution': CATEGORY_DISTRIBUTION,
            'model_used': generator.model
        },
        'entries': golden_entries
    }

    # Save to file
    output_path = Path(__file__).parent.parent / 'data' / 'golden_dataset.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(golden_dataset, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✓ Golden dataset saved to: {output_path}")
    logger.info(f"  Total entries: {len(golden_entries)}")

    # Summary by category
    category_counts = {}
    for entry in golden_entries:
        cat = entry['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1

    logger.info("\nCategory distribution:")
    for cat, count in category_counts.items():
        logger.info(f"  {cat}: {count}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ GOLDEN DATASET GENERATION COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
