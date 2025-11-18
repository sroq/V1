# Installation and Testing Guide

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/ss/Library/CloudStorage/OneDrive-Personal/Cubix/AI-asszisztens-fejlesztes/04-HF/V1

# Install the chunking pipeline dependencies
pip install -r chunking/requirements.txt
```

### 2. Configure API Key

Make sure your `.env` file contains a valid OpenAI API key:

```bash
# Edit .env file
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 3. Verify Database Setup

Ensure PostgreSQL with pgvector is running:

```bash
# Test database connection
psql -h localhost -p 5432 -U rag_user -d rag_assistant -c "\dt"
```

### 4. Run Validation

```bash
# Validate the pipeline setup
python chunking/chunker.py --input test_document.md --validate
```

## Testing the Pipeline

### Test 1: Load and Chunk Document (No Upload)

Test basic document loading and chunking without uploading to database:

```bash
python chunking/chunker.py \
    --input test_document.md \
    --strategy semantic \
    --log-level INFO
```

Expected output:
- Document loaded successfully
- Chunks generated
- Embeddings created (if API key is valid)
- No database upload
- Summary statistics printed

### Test 2: Process with Different Strategies

Test all four chunking strategies:

```bash
# Fixed-size chunking
python chunking/chunker.py \
    --input test_document.md \
    --strategy fixed \
    --chunk-size 512 \
    --chunk-overlap 50

# Semantic chunking
python chunking/chunker.py \
    --input test_document.md \
    --strategy semantic

# Recursive chunking
python chunking/chunker.py \
    --input test_document.md \
    --strategy recursive \
    --chunk-size 512

# Document-specific chunking
python chunking/chunker.py \
    --input test_document.md \
    --strategy document_specific
```

### Test 3: Full Pipeline with Upload

Process and upload to database:

```bash
python chunking/chunker.py \
    --input test_document.md \
    --strategy semantic \
    --upload \
    --log-level INFO
```

Expected output:
- Document loaded
- Chunks generated
- Embeddings created
- **Uploaded to database**
- Document ID returned

Verify in database:

```bash
psql -h localhost -p 5432 -U rag_user -d rag_assistant -c "SELECT COUNT(*) FROM document_chunks;"
```

### Test 4: Run Examples

Run the example script to see all features:

```bash
python chunking/example_usage.py
```

This will demonstrate:
- Programmatic usage
- CLI commands
- Strategy comparison
- Configuration examples
- Cost estimation

### Test 5: Process Multiple Documents

Create a test directory and process multiple files:

```bash
# Create test directory
mkdir -p test_docs

# Copy test document with different names
cp test_document.md test_docs/doc1.md
cp test_document.md test_docs/doc2.md

# Process all documents
python chunking/chunker.py \
    --input test_docs/ \
    --strategy semantic \
    --upload

# Check results
psql -h localhost -p 5432 -U rag_user -d rag_assistant -c "SELECT file_name, COUNT(*) as chunks FROM documents d JOIN document_chunks c ON d.id = c.document_id GROUP BY d.file_name;"
```

## Troubleshooting

### Issue: OpenAI API Key Error

```
Error: OPENAI_API_KEY environment variable not set
```

**Solution**:
1. Check `.env` file exists in project root
2. Verify it contains `OPENAI_API_KEY=sk-...`
3. Make sure `.env` is not in `.gitignore` excluded path

### Issue: Database Connection Failed

```
Error: Failed to connect to database
```

**Solution**:
1. Verify PostgreSQL is running: `pg_isready -h localhost -p 5432`
2. Test connection: `psql -h localhost -p 5432 -U rag_user -d rag_assistant`
3. Check credentials in `config.yaml` match your database

### Issue: pgvector Extension Missing

```
Error: pgvector extension is not installed
```

**Solution**:
```bash
psql -h localhost -p 5432 -U rag_user -d rag_assistant -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Issue: libmagic Warning

```
libmagic is unavailable but assists in filetype detection
```

**Solution** (optional):
```bash
# On macOS
brew install libmagic

# On Ubuntu/Debian
sudo apt-get install libmagic1
```

This warning is not critical - the pipeline will still work without libmagic.

### Issue: Rate Limit Errors

```
Error: Rate limit exceeded
```

**Solution**:
- Reduce `batch_size` in config.yaml:
  ```yaml
  embeddings:
    batch_size: 50  # Reduced from 100
  ```
- The pipeline will automatically retry with exponential backoff

## Verification Checklist

- [ ] Dependencies installed (`pip install -r chunking/requirements.txt`)
- [ ] `.env` file with valid OpenAI API key
- [ ] PostgreSQL running and accessible
- [ ] pgvector extension installed
- [ ] Database tables created (`documents`, `document_chunks`)
- [ ] Validation test passes (`--validate` flag)
- [ ] Can process test document without upload
- [ ] Can process and upload to database
- [ ] Can query chunks from database

## Next Steps

1. **Process Your Documents**: Point the pipeline to your actual documents
2. **Tune Configuration**: Adjust chunk sizes and strategies for your use case
3. **Monitor Costs**: Use cost estimation before large batches
4. **Build RAG System**: Use the vectorized chunks for semantic search

## Database Queries

### View Processed Documents

```sql
SELECT id, file_name, file_type, num_elements, processed_at
FROM documents
ORDER BY processed_at DESC;
```

### View Chunks for a Document

```sql
SELECT chunk_index, LEFT(content, 100) as preview, token_count
FROM document_chunks
WHERE document_id = 1
ORDER BY chunk_index;
```

### Count Total Chunks

```sql
SELECT COUNT(*) as total_chunks FROM document_chunks;
```

### Get Statistics

```sql
SELECT
    COUNT(DISTINCT document_id) as num_documents,
    COUNT(*) as num_chunks,
    AVG(token_count) as avg_tokens_per_chunk,
    SUM(token_count) as total_tokens
FROM document_chunks;
```

### Test Vector Similarity

```sql
-- Find similar chunks (requires an embedding to compare)
SELECT chunk_id, LEFT(content, 100) as preview
FROM document_chunks
ORDER BY embedding <-> (SELECT embedding FROM document_chunks WHERE id = 1)
LIMIT 5;
```

## Performance Tips

1. **Batch Processing**: Process multiple documents in one run for efficiency
2. **Progress Tracking**: Enable `save_progress` for large batches
3. **Appropriate Chunk Sizes**:
   - 256-512 tokens for precise retrieval
   - 512-1024 tokens for balanced performance
   - 1024+ tokens for more context
4. **Cost Management**: Use cost estimation before processing large document sets
5. **Database Indexes**: Ensure indexes are created on `document_chunks(document_id)` and `document_chunks.embedding`

## Support

If you encounter issues:
1. Check logs in `chunking_pipeline.log`
2. Run with `--log-level DEBUG` for detailed output
3. Review the troubleshooting section above
4. Check the main README.md for detailed documentation
