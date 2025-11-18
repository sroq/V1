# Chunking Pipeline - Quick Start Guide

## 1-Minute Setup

```bash
# 1. Install dependencies
pip install -r chunking/requirements.txt

# 2. Verify API key in .env
grep OPENAI_API_KEY .env

# 3. Test the pipeline
python chunking/chunker.py --input test_document.md --validate
```

## 5-Minute Tutorial

### Process Your First Document

```bash
# Process without uploading (safe to test)
python chunking/chunker.py \
    --input test_document.md \
    --strategy semantic \
    --log-level INFO
```

### Process and Upload to Database

```bash
# Process and upload to PostgreSQL
python chunking/chunker.py \
    --input test_document.md \
    --strategy semantic \
    --upload
```

### Verify Results

```bash
# Check database
psql -h localhost -p 5432 -U rag_user -d rag_assistant \
  -c "SELECT file_name, COUNT(*) as chunks FROM documents d JOIN document_chunks c ON d.id = c.document_id GROUP BY d.file_name;"
```

## Common Commands

### Process a Single File

```bash
python chunking/chunker.py --input document.pdf
```

### Process a Directory

```bash
python chunking/chunker.py --input /path/to/documents/
```

### Process with Custom Chunk Size

```bash
python chunking/chunker.py \
    --input documents/ \
    --chunk-size 1024 \
    --chunk-overlap 100 \
    --upload
```

### Process with Different Strategies

```bash
# Fixed-size chunking
python chunking/chunker.py --input docs/ --strategy fixed --upload

# Semantic chunking (default, preserves structure)
python chunking/chunker.py --input docs/ --strategy semantic --upload

# Recursive chunking (good for Markdown)
python chunking/chunker.py --input docs/ --strategy recursive --upload

# Document-specific (adapts to file type)
python chunking/chunker.py --input docs/ --strategy document_specific --upload
```

## Troubleshooting Quick Fixes

### OpenAI API Key Error

```bash
# Check if key is set
echo $OPENAI_API_KEY

# Or check .env file
cat .env | grep OPENAI_API_KEY

# Set it if missing
export OPENAI_API_KEY=sk-your-key-here
```

### Database Connection Error

```bash
# Test connection
psql -h localhost -p 5432 -U rag_user -d rag_assistant

# Check if PostgreSQL is running
pg_isready -h localhost -p 5432
```

### pgvector Extension Missing

```bash
# Install extension
psql -h localhost -p 5432 -U rag_user -d rag_assistant \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Strategy Selection Guide

| Document Type | Recommended Strategy | Reason |
|---------------|---------------------|--------|
| Blog posts, articles | `semantic` | Preserves article structure |
| Technical docs | `recursive` | Handles hierarchical headings |
| Research papers (PDF) | `document_specific` | Respects page boundaries |
| Code files | `document_specific` | Preserves syntax structure |
| Mixed types | `document_specific` | Adapts automatically |

## Chunk Size Guidelines

| Use Case | Chunk Size | Overlap |
|----------|------------|---------|
| Precise retrieval | 256 tokens | 25 tokens |
| Balanced (default) | 512 tokens | 50 tokens |
| More context | 1024 tokens | 100 tokens |
| Maximum context | 2048 tokens | 200 tokens |

## Examples

### Run All Examples

```bash
python chunking/example_usage.py
```

### Programmatic Usage

```python
from pathlib import Path
from chunking.loader import DocumentLoader
from chunking.strategies import SemanticChunker
from chunking.embeddings import EmbeddingGenerator

# Load document
loader = DocumentLoader()
doc = loader.load_document(Path("document.pdf"))

# Chunk document
chunker = SemanticChunker(max_chunk_size=1024)
chunks = chunker.chunk_document(doc)

# Generate embeddings
embedder = EmbeddingGenerator(api_key="your-key")
embeddings = embedder.generate_embeddings(chunks)
```

## File Locations

```
/Users/ss/Library/CloudStorage/OneDrive-Personal/Cubix/AI-asszisztens-fejlesztes/04-HF/V1/

├── chunking/
│   ├── chunker.py              # Main CLI
│   ├── loader.py              # Document loader
│   ├── strategies.py          # 4 chunking strategies
│   ├── embeddings.py          # OpenAI embeddings
│   ├── database.py            # PostgreSQL/pgvector
│   ├── utils.py               # Utilities
│   ├── config.yaml            # Configuration
│   ├── requirements.txt       # Dependencies
│   ├── README.md              # Full documentation
│   ├── INSTALLATION.md        # Installation guide
│   └── QUICKSTART.md          # This file
│
├── test_document.md           # Test file
├── .env                       # API keys
└── CHUNKING_PIPELINE_SUMMARY.md  # Implementation summary
```

## Help

```bash
# Show all command-line options
python chunking/chunker.py --help

# Run with debug logging
python chunking/chunker.py --input docs/ --log-level DEBUG
```

## Next Steps

1. Process your documents: `python chunking/chunker.py --input your_docs/`
2. Adjust chunk sizes: Try different `--chunk-size` values
3. Test strategies: Compare results with different `--strategy` options
4. Read full docs: Check `README.md` for comprehensive guide
5. Review summary: See `CHUNKING_PIPELINE_SUMMARY.md` for details

## Support

- Check logs: `cat chunking_pipeline.log`
- Review README: `chunking/README.md`
- Run examples: `python chunking/example_usage.py`
- Test setup: `python chunking/chunker.py --validate`
