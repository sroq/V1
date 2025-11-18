-- ============================================================================
-- RAG Assistant Database Initialization Script
-- PostgreSQL 15+ with pgvector extension
-- ============================================================================
--
-- This script initializes the database schema for a RAG-based AI assistant.
-- It includes:
-- - pgvector extension for vector similarity search
-- - Tables for document storage and embedding management
-- - HNSW indexes for high-performance vector similarity search
-- - Helper functions for vector operations
-- - Monitoring views for performance tracking
--
-- Vector Dimension: 1536 (OpenAI text-embedding-3-small)
-- Index Type: HNSW (Hierarchical Navigable Small World)
-- Distance Metric: Cosine similarity (using <=> operator)
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- ENUMS AND TYPES
-- ============================================================================

-- Document source types
CREATE TYPE source_type_enum AS ENUM (
    'pdf',
    'docx',
    'txt',
    'md',
    'html',
    'url',
    'api',
    'other'
);

-- Processing status for documents
CREATE TYPE processing_status_enum AS ENUM (
    'pending',
    'processing',
    'completed',
    'failed'
);

-- ============================================================================
-- MAIN TABLES
-- ============================================================================

-- Documents table: stores document metadata
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(500) NOT NULL,
    source_type source_type_enum NOT NULL,
    source_url TEXT,
    file_path TEXT,
    file_size_bytes BIGINT,
    file_hash VARCHAR(64), -- SHA-256 hash for deduplication
    processing_status processing_status_enum DEFAULT 'pending',
    processing_error TEXT,
    metadata JSONB DEFAULT '{}', -- Flexible metadata storage
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,

    -- Constraints
    CONSTRAINT unique_file_hash UNIQUE (file_hash),
    CONSTRAINT check_file_size CHECK (file_size_bytes IS NULL OR file_size_bytes > 0)
);

-- Create indexes for documents table
CREATE INDEX idx_documents_status ON documents(processing_status);
CREATE INDEX idx_documents_source_type ON documents(source_type);
CREATE INDEX idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX idx_documents_metadata ON documents USING GIN(metadata);

-- Document chunks table: stores text chunks with embeddings
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536), -- OpenAI text-embedding-3-small dimension
    token_count INTEGER,
    metadata JSONB DEFAULT '{}', -- Chunk-specific metadata (page number, section, etc.)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_document_chunk UNIQUE (document_id, chunk_index),
    CONSTRAINT check_chunk_index CHECK (chunk_index >= 0),
    CONSTRAINT check_token_count CHECK (token_count IS NULL OR token_count > 0),
    CONSTRAINT check_content_not_empty CHECK (LENGTH(content) > 0)
);

-- Create indexes for document_chunks table
CREATE INDEX idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX idx_chunks_created_at ON document_chunks(created_at DESC);
CREATE INDEX idx_chunks_metadata ON document_chunks USING GIN(metadata);

-- HNSW index for vector similarity search
-- Index parameters explanation:
-- - m=16: Number of bidirectional links per node (default: 16, range: 2-100)
--   Higher m = better recall but more memory and slower build time
--   16 is optimal for most use cases
-- - ef_construction=64: Size of dynamic candidate list during index construction
--   Higher ef_construction = better index quality but slower build time
--   64 is a good balance for 1536-dimensional vectors
--
-- IMPORTANT: Create this index AFTER loading your initial data for better performance
-- For now, we create it immediately to make the system ready to use
CREATE INDEX idx_chunks_embedding_hnsw ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Alternative: IVFFlat index (commented out, use if HNSW is too memory-intensive)
-- IVFFlat is faster to build but slower at query time
-- Lists = sqrt(rows) is a good starting point, adjust after loading data
-- CREATE INDEX idx_chunks_embedding_ivfflat ON document_chunks
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);

-- ============================================================================
-- SEARCH HISTORY AND ANALYTICS
-- ============================================================================

-- Search queries table: stores user queries and results for analytics
CREATE TABLE search_queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text TEXT NOT NULL,
    query_embedding vector(1536),
    results_count INTEGER,
    execution_time_ms NUMERIC(10, 2),
    user_id VARCHAR(255), -- Optional: if you have user authentication
    session_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT check_results_count CHECK (results_count >= 0),
    CONSTRAINT check_execution_time CHECK (execution_time_ms >= 0)
);

-- Create indexes for search_queries table
CREATE INDEX idx_queries_created_at ON search_queries(created_at DESC);
CREATE INDEX idx_queries_user_id ON search_queries(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX idx_queries_session_id ON search_queries(session_id) WHERE session_id IS NOT NULL;

-- Query results table: stores which chunks were returned for each query
CREATE TABLE query_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id UUID NOT NULL REFERENCES search_queries(id) ON DELETE CASCADE,
    chunk_id UUID NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    similarity_score NUMERIC(5, 4) NOT NULL,
    rank_position INTEGER NOT NULL,
    was_clicked BOOLEAN DEFAULT FALSE,
    was_helpful BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_query_chunk UNIQUE (query_id, chunk_id),
    CONSTRAINT check_rank_position CHECK (rank_position > 0),
    CONSTRAINT check_similarity_score CHECK (similarity_score >= 0 AND similarity_score <= 1)
);

-- Create indexes for query_results table
CREATE INDEX idx_query_results_query_id ON query_results(query_id);
CREATE INDEX idx_query_results_chunk_id ON query_results(chunk_id);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to update the updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update updated_at on documents table
CREATE TRIGGER trigger_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to perform vector similarity search with filters
CREATE OR REPLACE FUNCTION search_similar_chunks(
    query_embedding vector(1536),
    match_threshold numeric DEFAULT 0.7,
    match_count integer DEFAULT 10,
    filter_document_ids uuid[] DEFAULT NULL,
    filter_source_types source_type_enum[] DEFAULT NULL
)
RETURNS TABLE (
    chunk_id uuid,
    document_id uuid,
    document_title varchar,
    chunk_index integer,
    content text,
    similarity_score numeric,
    source_type source_type_enum,
    chunk_metadata jsonb
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id AS chunk_id,
        dc.document_id,
        d.title AS document_title,
        dc.chunk_index,
        dc.content,
        (1 - (dc.embedding <=> query_embedding)) AS similarity_score,
        d.source_type,
        dc.metadata AS chunk_metadata
    FROM document_chunks dc
    INNER JOIN documents d ON dc.document_id = d.id
    WHERE
        dc.embedding IS NOT NULL
        AND d.processing_status = 'completed'
        AND (1 - (dc.embedding <=> query_embedding)) >= match_threshold
        AND (filter_document_ids IS NULL OR dc.document_id = ANY(filter_document_ids))
        AND (filter_source_types IS NULL OR d.source_type = ANY(filter_source_types))
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to get statistics about the vector database
CREATE OR REPLACE FUNCTION get_database_stats()
RETURNS TABLE (
    total_documents bigint,
    total_chunks bigint,
    chunks_with_embeddings bigint,
    avg_chunks_per_document numeric,
    total_storage_mb numeric,
    pending_documents bigint,
    failed_documents bigint
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(DISTINCT d.id) AS total_documents,
        COUNT(dc.id) AS total_chunks,
        COUNT(dc.embedding) AS chunks_with_embeddings,
        ROUND(COUNT(dc.id)::numeric / NULLIF(COUNT(DISTINCT d.id), 0), 2) AS avg_chunks_per_document,
        ROUND(pg_database_size(current_database())::numeric / (1024 * 1024), 2) AS total_storage_mb,
        COUNT(DISTINCT d.id) FILTER (WHERE d.processing_status = 'pending') AS pending_documents,
        COUNT(DISTINCT d.id) FILTER (WHERE d.processing_status = 'failed') AS failed_documents
    FROM documents d
    LEFT JOIN document_chunks dc ON d.id = dc.document_id;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to clean up old search history (for data retention policies)
CREATE OR REPLACE FUNCTION cleanup_old_search_history(retention_days integer DEFAULT 90)
RETURNS integer AS $$
DECLARE
    deleted_count integer;
BEGIN
    DELETE FROM search_queries
    WHERE created_at < NOW() - (retention_days || ' days')::interval;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- MONITORING VIEWS
-- ============================================================================

-- View for monitoring document processing status
CREATE VIEW v_document_processing_status AS
SELECT
    processing_status,
    source_type,
    COUNT(*) AS document_count,
    SUM(file_size_bytes) AS total_size_bytes,
    ROUND(AVG(file_size_bytes), 0) AS avg_size_bytes
FROM documents
GROUP BY processing_status, source_type
ORDER BY processing_status, source_type;

-- View for vector search performance analytics
CREATE VIEW v_search_performance AS
SELECT
    DATE_TRUNC('hour', created_at) AS hour,
    COUNT(*) AS query_count,
    ROUND(AVG(execution_time_ms), 2) AS avg_execution_ms,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms), 2) AS p95_execution_ms,
    ROUND(AVG(results_count), 2) AS avg_results_count
FROM search_queries
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', created_at)
ORDER BY hour DESC;

-- View for most searched topics (based on query text frequency)
CREATE VIEW v_popular_queries AS
SELECT
    query_text,
    COUNT(*) AS query_count,
    ROUND(AVG(results_count), 2) AS avg_results,
    MAX(created_at) AS last_queried
FROM search_queries
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY query_text
HAVING COUNT(*) > 1
ORDER BY query_count DESC
LIMIT 100;

-- ============================================================================
-- INITIAL DATA AND CONFIGURATION
-- ============================================================================

-- Set default search configuration for full-text search (if needed in future)
ALTER DATABASE rag_assistant SET default_text_search_config = 'pg_catalog.english';

-- Grant appropriate permissions (adjust based on your security requirements)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO rag_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO rag_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO rag_user;

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE documents IS 'Stores document metadata and processing status';
COMMENT ON TABLE document_chunks IS 'Stores text chunks with their vector embeddings for similarity search';
COMMENT ON TABLE search_queries IS 'Logs all search queries for analytics and improvement';
COMMENT ON TABLE query_results IS 'Tracks which chunks were returned for each query';

COMMENT ON COLUMN document_chunks.embedding IS 'Vector embedding (1536 dimensions) generated by OpenAI text-embedding-3-small';
COMMENT ON COLUMN document_chunks.token_count IS 'Number of tokens in the chunk (useful for LLM context management)';
COMMENT ON COLUMN documents.file_hash IS 'SHA-256 hash of the file content for deduplication';

COMMENT ON FUNCTION search_similar_chunks IS 'Performs cosine similarity search with optional filters. Returns top-k most similar chunks.';
COMMENT ON FUNCTION get_database_stats IS 'Returns comprehensive statistics about the vector database';
COMMENT ON FUNCTION cleanup_old_search_history IS 'Removes search history older than specified retention period';

-- ============================================================================
-- MAINTENANCE RECOMMENDATIONS
-- ============================================================================

-- After loading initial data, consider:
-- 1. Running ANALYZE to update statistics: ANALYZE document_chunks;
-- 2. Monitoring index usage: SELECT * FROM pg_stat_user_indexes WHERE schemaname = 'public';
-- 3. Regular VACUUM to maintain performance: VACUUM ANALYZE document_chunks;
-- 4. Adjusting HNSW ef_search parameter at query time for better recall:
--    SET hnsw.ef_search = 100; (default: 40, higher = better recall but slower)

-- ============================================================================
-- SCRIPT COMPLETION
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Database initialization completed successfully!';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Created extensions: vector, uuid-ossp';
    RAISE NOTICE 'Created tables: documents, document_chunks, search_queries, query_results';
    RAISE NOTICE 'Created indexes: HNSW index for vector similarity search';
    RAISE NOTICE 'Created functions: search_similar_chunks, get_database_stats, cleanup_old_search_history';
    RAISE NOTICE 'Created views: v_document_processing_status, v_search_performance, v_popular_queries';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Vector dimension: 1536 (OpenAI text-embedding-3-small)';
    RAISE NOTICE 'Distance metric: Cosine similarity';
    RAISE NOTICE 'Index type: HNSW (m=16, ef_construction=64)';
    RAISE NOTICE '========================================';
END $$;
