#!/usr/bin/env python3
"""
Example usage of the RAG Assistant Chunking Pipeline.

Demonstrates both CLI and programmatic usage.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Example 1: Programmatic usage
def example_programmatic():
    """Example of programmatic pipeline usage."""
    print("=" * 60)
    print("Example 1: Programmatic Usage")
    print("=" * 60)

    from chunking.loader import DocumentLoader
    from chunking.strategies import SemanticChunker
    from chunking.embeddings import EmbeddingGenerator
    from chunking.database import DatabaseUploader

    # Create a test document
    test_doc_path = Path("test_document.md")
    test_content = """# Sample Document

This is a sample document for testing the chunking pipeline.

## Introduction

The RAG Assistant Chunking Pipeline provides comprehensive document processing
capabilities for AI applications.

## Features

- Multiple chunking strategies
- OpenAI embeddings integration
- PostgreSQL/pgvector storage
- Progress tracking and logging

## Conclusion

This pipeline is production-ready and extensible.
"""
    test_doc_path.write_text(test_content)

    try:
        # 1. Load document
        print("\n1. Loading document...")
        loader = DocumentLoader(extract_metadata="rich")
        doc = loader.load_document(test_doc_path)

        if doc:
            print(f"   Loaded: {doc.metadata.file_name}")
            print(f"   Elements: {len(doc.elements)}")
            print(f"   Structure: {doc.metadata.custom_metadata.get('structure')}")

            # 2. Chunk document
            print("\n2. Chunking document...")
            chunker = SemanticChunker(max_chunk_size=512)
            chunks = chunker.chunk_document(doc)

            print(f"   Generated {len(chunks)} chunks")
            for i, chunk in enumerate(chunks[:3], 1):  # Show first 3
                print(f"   Chunk {i}: {len(chunk.content)} chars, {chunk.token_count} tokens")

            # 3. Generate embeddings
            print("\n3. Generating embeddings...")
            api_key = os.getenv('OPENAI_API_KEY')

            if not api_key:
                print("   OPENAI_API_KEY not set, skipping embedding generation")
                print("   Set your API key in .env file to test this feature")
            else:
                embedder = EmbeddingGenerator(api_key=api_key)

                # Estimate cost first
                cost_info = embedder.estimate_cost(len(chunks))
                print(f"   Estimated cost: ${cost_info['estimated_cost_usd']:.4f}")

                # Generate embeddings (for first chunk only in this example)
                embeddings = embedder.generate_embeddings(chunks[:1], show_progress=False)
                print(f"   Generated {len(embeddings)} embeddings")
                if embeddings:
                    print(f"   Embedding dimensions: {embeddings[0].dimensions}")

            # 4. Database upload (optional)
            print("\n4. Database upload...")
            print("   Skipping database upload in this example")
            print("   Use --upload flag with CLI to upload to database")

    finally:
        # Cleanup
        if test_doc_path.exists():
            test_doc_path.unlink()

    print("\n" + "=" * 60 + "\n")


# Example 2: CLI usage examples
def example_cli_commands():
    """Print example CLI commands."""
    print("=" * 60)
    print("Example 2: CLI Usage Commands")
    print("=" * 60)

    examples = [
        (
            "Process a single file with semantic chunking",
            "python chunking/chunker.py --input document.pdf --strategy semantic"
        ),
        (
            "Process directory with fixed-size chunking and upload",
            "python chunking/chunker.py --input docs/ --strategy fixed --chunk-size 512 --upload"
        ),
        (
            "Use custom config file",
            "python chunking/chunker.py --input docs/ --config my_config.yaml --upload"
        ),
        (
            "Validate setup before processing",
            "python chunking/chunker.py --input docs/ --validate"
        ),
        (
            "Process with recursive chunking",
            "python chunking/chunker.py --input docs/ --strategy recursive --chunk-overlap 50"
        ),
        (
            "Process with debug logging",
            "python chunking/chunker.py --input docs/ --log-level DEBUG"
        ),
        (
            "Clear progress and restart",
            "python chunking/chunker.py --input docs/ --clear-progress --upload"
        ),
    ]

    for i, (description, command) in enumerate(examples, 1):
        print(f"\n{i}. {description}")
        print(f"   {command}")

    print("\n" + "=" * 60 + "\n")


# Example 3: Strategy comparison
def example_strategy_comparison():
    """Compare different chunking strategies."""
    print("=" * 60)
    print("Example 3: Strategy Comparison")
    print("=" * 60)

    from chunking.loader import DocumentLoader
    from chunking.strategies import (
        FixedSizeChunker,
        SemanticChunker,
        RecursiveChunker,
        DocumentTypeSpecificChunker
    )

    # Create a test document
    test_doc_path = Path("test_comparison.md")
    test_content = """# Document Title

## Section 1: Introduction

This is the introduction section with some content.
It has multiple sentences. Each sentence adds context.

## Section 2: Main Content

### Subsection 2.1

First subsection content goes here.

### Subsection 2.2

Second subsection content goes here.

## Section 3: Conclusion

Final thoughts and summary.
"""
    test_doc_path.write_text(test_content)

    try:
        # Load document
        loader = DocumentLoader()
        doc = loader.load_document(test_doc_path)

        if not doc:
            print("Failed to load document")
            return

        # Test different strategies
        strategies = [
            ("Fixed-Size (256 tokens)", FixedSizeChunker(chunk_size=256, chunk_overlap=25)),
            ("Semantic (512 tokens)", SemanticChunker(max_chunk_size=512)),
            ("Recursive (256 tokens)", RecursiveChunker(chunk_size=256, chunk_overlap=25)),
            ("Document-Specific", DocumentTypeSpecificChunker()),
        ]

        print(f"\nOriginal document: {len(test_content)} chars, ~{len(test_content)//4} tokens\n")

        for name, chunker in strategies:
            chunks = chunker.chunk_document(doc)

            print(f"{name}:")
            print(f"  Total chunks: {len(chunks)}")

            if chunks:
                avg_tokens = sum(c.token_count for c in chunks) / len(chunks)
                print(f"  Average tokens per chunk: {avg_tokens:.0f}")

                # Show first chunk preview
                first_chunk = chunks[0].content[:100].replace('\n', ' ')
                print(f"  First chunk preview: {first_chunk}...")

            print()

    finally:
        # Cleanup
        if test_doc_path.exists():
            test_doc_path.unlink()

    print("=" * 60 + "\n")


# Example 4: Configuration examples
def example_configurations():
    """Show example configurations."""
    print("=" * 60)
    print("Example 4: Configuration Examples")
    print("=" * 60)

    configs = {
        "Small Documents (Blog Posts, Articles)": {
            "strategy": "semantic",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "reason": "Preserves article structure, good for precise retrieval"
        },
        "Large Technical Documents (Manuals, Specs)": {
            "strategy": "recursive",
            "chunk_size": 1024,
            "chunk_overlap": 100,
            "reason": "Handles hierarchical structure, maintains context"
        },
        "Mixed Document Types": {
            "strategy": "document_specific",
            "reason": "Adapts to each document type automatically"
        },
        "Code Documentation": {
            "strategy": "recursive",
            "chunk_size": 768,
            "chunk_overlap": 75,
            "reason": "Preserves code structure and function boundaries"
        },
        "Research Papers (PDFs)": {
            "strategy": "document_specific",
            "chunk_size": 1024,
            "reason": "Respects page boundaries and extracts tables"
        },
    }

    for use_case, config in configs.items():
        print(f"\n{use_case}:")
        print(f"  Strategy: {config['strategy']}")
        if 'chunk_size' in config:
            print(f"  Chunk size: {config['chunk_size']} tokens")
        if 'chunk_overlap' in config:
            print(f"  Overlap: {config['chunk_overlap']} tokens")
        print(f"  Reason: {config['reason']}")

    print("\n" + "=" * 60 + "\n")


# Example 5: Cost estimation
def example_cost_estimation():
    """Demonstrate cost estimation."""
    print("=" * 60)
    print("Example 5: Cost Estimation")
    print("=" * 60)

    from chunking.embeddings import EmbeddingGenerator

    api_key = os.getenv('OPENAI_API_KEY', 'dummy-key-for-estimation')
    embedder = EmbeddingGenerator(api_key=api_key)

    scenarios = [
        (100, "Small project (100 chunks)"),
        (1000, "Medium project (1,000 chunks)"),
        (10000, "Large project (10,000 chunks)"),
        (100000, "Enterprise project (100,000 chunks)"),
    ]

    print("\nOpenAI text-embedding-3-small cost estimates:")
    print("-" * 60)

    for num_chunks, description in scenarios:
        cost_info = embedder.estimate_cost(num_chunks)
        print(f"\n{description}")
        print(f"  Chunks: {cost_info['num_chunks']:,}")
        print(f"  Estimated tokens: {cost_info['estimated_tokens']:,}")
        print(f"  Estimated cost: ${cost_info['estimated_cost_usd']:.4f}")

    print("\n" + "=" * 60 + "\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("RAG ASSISTANT CHUNKING PIPELINE - EXAMPLES")
    print("=" * 60 + "\n")

    examples = [
        ("Programmatic Usage", example_programmatic),
        ("CLI Commands", example_cli_commands),
        ("Strategy Comparison", example_strategy_comparison),
        ("Configuration Examples", example_configurations),
        ("Cost Estimation", example_cost_estimation),
    ]

    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\nError running {name}: {e}\n")

    print("=" * 60)
    print("EXAMPLES COMPLETE")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
