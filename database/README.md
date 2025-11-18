# PostgreSQL Vector Database for RAG Assistant

This directory contains the PostgreSQL database configuration with pgvector extension for the RAG-based AI assistant system.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Database Schema](#database-schema)
- [Vector Search](#vector-search)
- [Usage Examples](#usage-examples)
- [Performance Optimization](#performance-optimization)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Backup and Recovery](#backup-and-recovery)
- [Troubleshooting](#troubleshooting)

## Overview

### Technology Stack

- **Database**: PostgreSQL 16
- **Extension**: pgvector 0.5+
- **Vector Dimension**: 1536 (OpenAI text-embedding-3-small)
- **Index Type**: HNSW (Hierarchical Navigable Small World)
- **Distance Metric**: Cosine similarity

### Key Features

- ✅ High-performance vector similarity search using HNSW indexing
- ✅ Document storage with metadata and processing status tracking
- ✅ Text chunking with configurable overlap
- ✅ Search analytics and query history
- ✅ Automatic timestamp tracking
- ✅ Flexible metadata storage with JSONB
- ✅ Production-ready configuration with health checks
- ✅ Optional PgAdmin interface for database management

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum (8GB recommended)
- 10GB disk space minimum

## Quick Start

### 1. Initial Setup

```bash
# Clone or navigate to the project directory
cd /Users/ss/Library/CloudStorage/OneDrive-Personal/Cubix/AI-asszisztens-fejlesztes/04-HF/V1

# Copy the environment template
cp .env.example .env

# Edit .env and set your passwords
nano .env  # or use your preferred editor
```

**Important**: Change these values in `.env`:
- `POSTGRES_PASSWORD` - Set a strong password
- `PGADMIN_PASSWORD` - Set admin password (if using PgAdmin)

### 2. Start the Database

```bash
# Start PostgreSQL only
docker-compose up -d

# Or start with PgAdmin for database management UI
docker-compose --profile admin up -d
```

### 3. Verify Installation

```bash
# Check if container is running
docker-compose ps

# Check container logs
docker-compose logs postgres

# Verify database is healthy
docker-compose exec postgres pg_isready -U rag_user -d rag_assistant
```

### 4. Connect to the Database

```bash
# Using psql inside the container
docker-compose exec postgres psql -U rag_user -d rag_assistant

# Using psql from your host (if PostgreSQL client is installed)
psql -h localhost -p 5432 -U rag_user -d rag_assistant
```

### 5. Verify pgvector Extension

```sql
-- Check pgvector version
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Test vector operations
SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector AS euclidean_distance;

-- Check database statistics
SELECT * FROM get_database_stats();
```

## Database Schema

### Core Tables

#### `documents`
Stores document metadata and processing status.

```sql
Column              | Type                     | Description
--------------------|--------------------------|----------------------------------
id                  | UUID                     | Primary key (auto-generated)
title               | VARCHAR(500)             | Document title
source_type         | source_type_enum         | Type: pdf, docx, txt, md, html, url, api, other
source_url          | TEXT                     | Original URL (if applicable)
file_path           | TEXT                     | Local file path
file_size_bytes     | BIGINT                   | File size in bytes
file_hash           | VARCHAR(64)              | SHA-256 hash for deduplication
processing_status   | processing_status_enum   | Status: pending, processing, completed, failed
processing_error    | TEXT                     | Error message if processing failed
metadata            | JSONB                    | Flexible metadata storage
created_at          | TIMESTAMP WITH TIME ZONE | Creation timestamp
updated_at          | TIMESTAMP WITH TIME ZONE | Last update timestamp (auto-updated)
processed_at        | TIMESTAMP WITH TIME ZONE | Processing completion timestamp
```

#### `document_chunks`
Stores text chunks with their vector embeddings.

```sql
Column       | Type                     | Description
-------------|--------------------------|------------------------------------------
id           | UUID                     | Primary key (auto-generated)
document_id  | UUID                     | Foreign key to documents table
chunk_index  | INTEGER                  | Position in document (0-based)
content      | TEXT                     | Full text of the chunk
embedding    | vector(1536)             | Vector embedding (OpenAI text-embedding-3-small)
token_count  | INTEGER                  | Number of tokens in chunk
metadata     | JSONB                    | Chunk-specific metadata (page, section, etc.)
created_at   | TIMESTAMP WITH TIME ZONE | Creation timestamp
```

**Indexes:**
- HNSW index on `embedding` for fast similarity search
- B-tree indexes on `document_id`, `created_at`, and `metadata`

#### `search_queries`
Logs all search queries for analytics.

```sql
Column            | Type                     | Description
------------------|--------------------------|----------------------------------
id                | UUID                     | Primary key (auto-generated)
query_text        | TEXT                     | Original search query text
query_embedding   | vector(1536)             | Query embedding vector
results_count     | INTEGER                  | Number of results returned
execution_time_ms | NUMERIC(10,2)            | Query execution time in milliseconds
user_id           | VARCHAR(255)             | User identifier (optional)
session_id        | VARCHAR(255)             | Session identifier
metadata          | JSONB                    | Additional query metadata
created_at        | TIMESTAMP WITH TIME ZONE | Query timestamp
```

#### `query_results`
Tracks which chunks were returned for each query.

```sql
Column           | Type                     | Description
-----------------|--------------------------|----------------------------------
id               | UUID                     | Primary key (auto-generated)
query_id         | UUID                     | Foreign key to search_queries
chunk_id         | UUID                     | Foreign key to document_chunks
similarity_score | NUMERIC(5,4)             | Cosine similarity score (0-1)
rank_position    | INTEGER                  | Position in search results (1-based)
was_clicked      | BOOLEAN                  | Whether user clicked the result
was_helpful      | BOOLEAN                  | User feedback on result quality
created_at       | TIMESTAMP WITH TIME ZONE | Result timestamp
```

### Helper Functions

#### `search_similar_chunks()`
Performs vector similarity search with optional filters.

```sql
-- Function signature
search_similar_chunks(
    query_embedding vector(1536),
    match_threshold numeric DEFAULT 0.7,
    match_count integer DEFAULT 10,
    filter_document_ids uuid[] DEFAULT NULL,
    filter_source_types source_type_enum[] DEFAULT NULL
)

-- Returns:
-- chunk_id, document_id, document_title, chunk_index, content,
-- similarity_score, source_type, chunk_metadata
```

#### `get_database_stats()`
Returns comprehensive database statistics.

```sql
-- Returns:
-- total_documents, total_chunks, chunks_with_embeddings,
-- avg_chunks_per_document, total_storage_mb, pending_documents, failed_documents
```

#### `cleanup_old_search_history()`
Removes old search history based on retention policy.

```sql
-- Function signature
cleanup_old_search_history(retention_days integer DEFAULT 90)
-- Returns: number of deleted records
```

## Vector Search

### Understanding Distance Metrics

The database uses **cosine similarity** as the distance metric, implemented with the `<=>` operator:

- **Cosine Similarity**: Measures the angle between vectors (0 = identical, 1 = opposite)
- Best for: Text embeddings where direction matters more than magnitude
- Returns: Values between 0 (most similar) and 2 (most dissimilar)

**Conversion to similarity score**: `similarity = 1 - distance`

### HNSW Index Configuration

Current configuration:
- **m = 16**: Number of bidirectional links (optimal for most use cases)
- **ef_construction = 64**: Build-time quality parameter (good balance)
- **ef_search = 40**: Query-time recall parameter (can be adjusted)

### Tuning Recall vs Speed

At query time, you can adjust the `ef_search` parameter:

```sql
-- Higher ef_search = better recall but slower queries
SET hnsw.ef_search = 100;  -- Better recall
SET hnsw.ef_search = 200;  -- Even better recall (slower)
SET hnsw.ef_search = 40;   -- Default (faster)
```

## Usage Examples

### 1. Insert a Document

```sql
-- Insert a new document
INSERT INTO documents (title, source_type, file_path, file_size_bytes, file_hash, metadata)
VALUES (
    'Machine Learning Best Practices',
    'pdf',
    '/data/documents/ml-practices.pdf',
    1024576,
    'a7f3b2c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1',
    '{"author": "John Doe", "year": 2024, "category": "AI"}'
)
RETURNING id;

-- Update processing status
UPDATE documents
SET processing_status = 'completed', processed_at = NOW()
WHERE id = 'your-document-id';
```

### 2. Insert Document Chunks with Embeddings

```sql
-- Insert a chunk with embedding
-- Note: In practice, you'll generate embeddings using OpenAI API
INSERT INTO document_chunks (document_id, chunk_index, content, embedding, token_count, metadata)
VALUES (
    'your-document-id',
    0,
    'Machine learning is a subset of artificial intelligence...',
    '[0.1, 0.2, 0.3, ...]',  -- 1536-dimensional vector from OpenAI
    150,
    '{"page": 1, "section": "Introduction"}'
);
```

### 3. Perform Vector Similarity Search

#### Basic Search

```sql
-- Search for similar chunks using the helper function
SELECT * FROM search_similar_chunks(
    '[0.1, 0.2, 0.3, ...]',  -- Your query embedding
    0.7,                      -- Minimum similarity threshold
    10                        -- Number of results
);
```

#### Filtered Search

```sql
-- Search within specific documents
SELECT * FROM search_similar_chunks(
    '[0.1, 0.2, 0.3, ...]',
    0.7,
    10,
    ARRAY['doc-id-1', 'doc-id-2']::uuid[],  -- Filter by document IDs
    NULL
);

-- Search within specific document types
SELECT * FROM search_similar_chunks(
    '[0.1, 0.2, 0.3, ...]',
    0.7,
    10,
    NULL,
    ARRAY['pdf', 'docx']::source_type_enum[]  -- Filter by source types
);
```

#### Raw Vector Search (without helper function)

```sql
-- Direct vector search with custom query
SELECT
    dc.id,
    dc.content,
    d.title,
    (1 - (dc.embedding <=> '[0.1, 0.2, ...]'::vector)) AS similarity_score
FROM document_chunks dc
INNER JOIN documents d ON dc.document_id = d.id
WHERE
    dc.embedding IS NOT NULL
    AND (1 - (dc.embedding <=> '[0.1, 0.2, ...]'::vector)) >= 0.7
ORDER BY dc.embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;
```

### 4. Log Search Queries

```sql
-- Insert a search query log
INSERT INTO search_queries (query_text, query_embedding, results_count, execution_time_ms, session_id)
VALUES (
    'What are machine learning best practices?',
    '[0.1, 0.2, ...]',
    10,
    45.23,
    'session-123'
)
RETURNING id;

-- Insert query results
INSERT INTO query_results (query_id, chunk_id, similarity_score, rank_position)
VALUES
    ('query-id', 'chunk-id-1', 0.95, 1),
    ('query-id', 'chunk-id-2', 0.89, 2),
    ('query-id', 'chunk-id-3', 0.85, 3);
```

### 5. Analytics Queries

```sql
-- Get database statistics
SELECT * FROM get_database_stats();

-- View document processing status
SELECT * FROM v_document_processing_status;

-- View search performance metrics
SELECT * FROM v_search_performance;

-- Find most popular queries
SELECT * FROM v_popular_queries LIMIT 20;

-- Get average similarity scores by source type
SELECT
    d.source_type,
    COUNT(*) AS search_count,
    ROUND(AVG(qr.similarity_score), 4) AS avg_similarity,
    ROUND(MIN(qr.similarity_score), 4) AS min_similarity,
    ROUND(MAX(qr.similarity_score), 4) AS max_similarity
FROM query_results qr
JOIN document_chunks dc ON qr.chunk_id = dc.id
JOIN documents d ON dc.document_id = d.id
GROUP BY d.source_type
ORDER BY search_count DESC;
```

## Performance Optimization

### Initial Data Loading

When loading large amounts of data:

1. **Drop indexes before bulk insert** (much faster):
```sql
DROP INDEX IF EXISTS idx_chunks_embedding_hnsw;
```

2. **Load your data**:
```sql
-- Insert thousands of chunks...
```

3. **Recreate indexes after loading**:
```sql
CREATE INDEX idx_chunks_embedding_hnsw ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

4. **Update statistics**:
```sql
ANALYZE document_chunks;
```

### Query Optimization

```sql
-- Use EXPLAIN ANALYZE to understand query performance
EXPLAIN ANALYZE
SELECT * FROM search_similar_chunks('[0.1, 0.2, ...]', 0.7, 10);

-- Adjust ef_search for better recall
SET hnsw.ef_search = 100;

-- Check index usage
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;
```

### Connection Pooling

For production applications, use connection pooling:

**Python (SQLAlchemy)**:
```python
from sqlalchemy import create_engine

engine = create_engine(
    'postgresql://rag_user:password@localhost:5432/rag_assistant',
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)
```

**Node.js (pg)**:
```javascript
const { Pool } = require('pg');

const pool = new Pool({
    host: 'localhost',
    port: 5432,
    database: 'rag_assistant',
    user: 'rag_user',
    password: 'password',
    max: 20,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 2000,
});
```

### Regular Maintenance

```sql
-- Vacuum analyze (run periodically, e.g., daily)
VACUUM ANALYZE document_chunks;
VACUUM ANALYZE documents;

-- Reindex (run if query performance degrades)
REINDEX INDEX CONCURRENTLY idx_chunks_embedding_hnsw;

-- Clean up old search history (run weekly)
SELECT cleanup_old_search_history(90);  -- Keep last 90 days
```

## Monitoring and Maintenance

### Health Checks

```bash
# Check database health
docker-compose exec postgres pg_isready -U rag_user

# Check database size
docker-compose exec postgres psql -U rag_user -d rag_assistant -c "
SELECT pg_size_pretty(pg_database_size('rag_assistant')) AS database_size;
"

# Check table sizes
docker-compose exec postgres psql -U rag_user -d rag_assistant -c "
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"
```

### Performance Monitoring

```sql
-- Check slow queries (from logs)
SELECT
    query,
    calls,
    total_time,
    mean_time,
    max_time
FROM pg_stat_statements
WHERE query LIKE '%document_chunks%'
ORDER BY mean_time DESC
LIMIT 10;

-- Check cache hit ratio (should be > 99%)
SELECT
    sum(heap_blks_read) AS heap_read,
    sum(heap_blks_hit)  AS heap_hit,
    round(sum(heap_blks_hit) / nullif(sum(heap_blks_hit) + sum(heap_blks_read), 0) * 100, 2) AS cache_hit_ratio
FROM pg_statio_user_tables;

-- Check index usage efficiency
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;
```

### Automated Monitoring Script

Create a monitoring script to check database health:

```bash
#!/bin/bash
# monitoring.sh

echo "=== Database Health Check ==="
docker-compose exec -T postgres psql -U rag_user -d rag_assistant -c "SELECT * FROM get_database_stats();"

echo -e "\n=== Table Sizes ==="
docker-compose exec -T postgres psql -U rag_user -d rag_assistant -c "
SELECT
    tablename,
    pg_size_pretty(pg_total_relation_size('public.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size('public.'||tablename) DESC;
"

echo -e "\n=== Cache Hit Ratio ==="
docker-compose exec -T postgres psql -U rag_user -d rag_assistant -c "
SELECT
    round(sum(heap_blks_hit) / nullif(sum(heap_blks_hit) + sum(heap_blks_read), 0) * 100, 2) AS cache_hit_ratio
FROM pg_statio_user_tables;
"

echo -e "\n=== Recent Search Performance ==="
docker-compose exec -T postgres psql -U rag_user -d rag_assistant -c "SELECT * FROM v_search_performance LIMIT 5;"
```

## Backup and Recovery

### Manual Backup

```bash
# Full database backup
docker-compose exec postgres pg_dump -U rag_user -d rag_assistant -F c -f /tmp/backup.dump

# Copy backup from container to host
docker cp rag-postgres:/tmp/backup.dump ./backups/rag_assistant_$(date +%Y%m%d_%H%M%S).dump

# Backup only schema (no data)
docker-compose exec postgres pg_dump -U rag_user -d rag_assistant --schema-only > schema_backup.sql
```

### Automated Backup Script

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/rag_assistant_$TIMESTAMP.dump"

mkdir -p $BACKUP_DIR

echo "Starting backup at $TIMESTAMP..."
docker-compose exec -T postgres pg_dump -U rag_user -d rag_assistant -F c > $BACKUP_FILE

if [ $? -eq 0 ]; then
    echo "Backup completed successfully: $BACKUP_FILE"
    # Optional: Compress the backup
    gzip $BACKUP_FILE
    echo "Backup compressed: $BACKUP_FILE.gz"

    # Optional: Delete backups older than 30 days
    find $BACKUP_DIR -name "*.dump.gz" -mtime +30 -delete
    echo "Old backups cleaned up"
else
    echo "Backup failed!"
    exit 1
fi
```

### Restore from Backup

```bash
# Stop the application
docker-compose down

# Remove old data
rm -rf postgres-data/*

# Start PostgreSQL
docker-compose up -d postgres

# Wait for PostgreSQL to be ready
sleep 10

# Restore from backup
docker cp ./backups/rag_assistant_YYYYMMDD_HHMMSS.dump rag-postgres:/tmp/backup.dump
docker-compose exec postgres pg_restore -U rag_user -d rag_assistant -c /tmp/backup.dump

echo "Restore completed"
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs postgres

# Common issues:
# 1. Port 5432 already in use
#    Solution: Change POSTGRES_PORT in .env

# 2. Permission issues with postgres-data
#    Solution: Fix permissions
sudo chown -R 999:999 postgres-data/

# 3. Corrupted data directory
#    Solution: Remove and recreate
docker-compose down -v
rm -rf postgres-data
docker-compose up -d
```

### pgvector Extension Not Found

```bash
# Verify extension is available
docker-compose exec postgres psql -U rag_user -d rag_assistant -c "
SELECT * FROM pg_available_extensions WHERE name = 'vector';
"

# If not found, you may be using wrong image
# Ensure docker-compose.yml uses: pgvector/pgvector:pg16
```

### Slow Vector Search Queries

```sql
-- 1. Check if HNSW index is being used
EXPLAIN ANALYZE
SELECT * FROM search_similar_chunks('[0.1, 0.2, ...]', 0.7, 10);
-- Look for "Index Scan using idx_chunks_embedding_hnsw"

-- 2. Increase ef_search for better results
SET hnsw.ef_search = 100;

-- 3. Update statistics
ANALYZE document_chunks;

-- 4. Check for index bloat
SELECT
    pg_size_pretty(pg_relation_size('idx_chunks_embedding_hnsw')) AS index_size;

-- If index is bloated, reindex:
REINDEX INDEX CONCURRENTLY idx_chunks_embedding_hnsw;
```

### Out of Memory Errors

```sql
-- Reduce work_mem temporarily
SET work_mem = '16MB';

-- Or modify docker-compose.yml to allocate more memory
# deploy:
#   resources:
#     limits:
#       memory: 8G  # Increase from 4G
```

### Connection Pool Exhausted

```bash
# Check current connections
docker-compose exec postgres psql -U rag_user -d rag_assistant -c "
SELECT
    count(*) AS total_connections,
    count(*) FILTER (WHERE state = 'active') AS active_connections,
    count(*) FILTER (WHERE state = 'idle') AS idle_connections
FROM pg_stat_activity
WHERE datname = 'rag_assistant';
"

# Increase max_connections in docker-compose.yml if needed
# command: >
#   postgres
#   -c max_connections=200  # Add this line
```

### Vector Dimension Mismatch

```
ERROR: expected 1536 dimensions, got 768
```

**Solution**: Ensure you're using the correct embedding model:
- OpenAI `text-embedding-3-small`: 1536 dimensions ✓
- OpenAI `text-embedding-3-large`: 3072 dimensions ✗
- OpenAI `text-embedding-ada-002`: 1536 dimensions ✓

To change dimensions, you need to:
1. Drop the existing table
2. Modify `init.sql` to use the new dimension
3. Recreate the database

## Additional Resources

### PgAdmin Access

If you started the database with the admin profile:

```bash
docker-compose --profile admin up -d
```

Access PgAdmin at: `http://localhost:5050`

**Connection details**:
- Host: `postgres` (container name)
- Port: `5432`
- Database: `rag_assistant`
- Username: `rag_user`
- Password: (from your .env file)

### Useful Commands

```bash
# View running containers
docker-compose ps

# View logs (follow mode)
docker-compose logs -f postgres

# Stop the database
docker-compose down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose down -v

# Restart the database
docker-compose restart postgres

# Execute SQL file
docker-compose exec postgres psql -U rag_user -d rag_assistant -f /path/to/file.sql

# Export query results to CSV
docker-compose exec postgres psql -U rag_user -d rag_assistant -c "
COPY (SELECT * FROM documents) TO '/tmp/documents.csv' WITH CSV HEADER;
"
docker cp rag-postgres:/tmp/documents.csv ./documents.csv
```

### Performance Benchmarking

```sql
-- Benchmark vector search performance
DO $$
DECLARE
    start_time timestamp;
    end_time timestamp;
    query_time numeric;
    i integer;
BEGIN
    FOR i IN 1..10 LOOP
        start_time := clock_timestamp();

        PERFORM * FROM search_similar_chunks(
            (SELECT embedding FROM document_chunks LIMIT 1),
            0.7,
            10
        );

        end_time := clock_timestamp();
        query_time := EXTRACT(MILLISECONDS FROM (end_time - start_time));

        RAISE NOTICE 'Query % completed in % ms', i, query_time;
    END LOOP;
END $$;
```

## Support and Contributing

For issues, questions, or contributions:
1. Check the [PostgreSQL documentation](https://www.postgresql.org/docs/)
2. Check the [pgvector documentation](https://github.com/pgvector/pgvector)
3. Review the troubleshooting section above
4. Check container logs: `docker-compose logs postgres`

## License

This database configuration is part of the RAG Assistant project and follows the same license as the main project.

---

**Last Updated**: 2025-11-01
**PostgreSQL Version**: 16
**pgvector Version**: 0.5+
