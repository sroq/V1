-- ============================================================================
-- PostgreSQL Configuration for Vector Operations
-- ============================================================================
--
-- This script applies runtime configuration optimized for vector operations.
-- These settings complement the PostgreSQL configuration in docker-compose.yml
-- ============================================================================

-- Set application name for connection tracking
SET application_name = 'rag_assistant';

-- ============================================================================
-- VECTOR SEARCH PERFORMANCE TUNING
-- ============================================================================

-- HNSW-specific parameter: Controls the size of the dynamic candidate list
-- during search operations. Higher values = better recall but slower queries.
-- Default: 40
-- Recommended range: 40-200 depending on recall requirements
-- You can override this at query time with: SET hnsw.ef_search = 100;
ALTER DATABASE rag_assistant SET hnsw.ef_search = 40;

-- ============================================================================
-- QUERY PERFORMANCE SETTINGS
-- ============================================================================

-- Enable parallel query execution for vector operations
ALTER DATABASE rag_assistant SET max_parallel_workers_per_gather = 2;
ALTER DATABASE rag_assistant SET parallel_setup_cost = 1000;
ALTER DATABASE rag_assistant SET parallel_tuple_cost = 0.1;

-- Set work memory for sorting and hash operations
-- This affects ORDER BY and JOIN operations on large result sets
ALTER DATABASE rag_assistant SET work_mem = '32MB';

-- ============================================================================
-- MAINTENANCE AND AUTOVACUUM
-- ============================================================================

-- Configure autovacuum for better performance with frequent inserts/updates
ALTER DATABASE rag_assistant SET autovacuum_vacuum_scale_factor = 0.1;
ALTER DATABASE rag_assistant SET autovacuum_analyze_scale_factor = 0.05;

-- More aggressive autovacuum for tables with heavy write operations
ALTER TABLE document_chunks SET (
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_analyze_scale_factor = 0.02
);

ALTER TABLE documents SET (
    autovacuum_vacuum_scale_factor = 0.1,
    autovacuum_analyze_scale_factor = 0.05
);

-- ============================================================================
-- CONNECTION AND TIMEOUT SETTINGS
-- ============================================================================

-- Set statement timeout to prevent long-running queries from blocking
-- Adjust based on your expected query execution times
ALTER DATABASE rag_assistant SET statement_timeout = '60s';

-- Set idle transaction timeout to clean up abandoned connections
ALTER DATABASE rag_assistant SET idle_in_transaction_session_timeout = '10min';

-- ============================================================================
-- LOGGING AND MONITORING
-- ============================================================================

-- Log slow queries for performance monitoring
ALTER DATABASE rag_assistant SET log_min_duration_statement = '1000'; -- Log queries > 1 second

-- Log autovacuum activity for maintenance monitoring
ALTER DATABASE rag_assistant SET log_autovacuum_min_duration = '0';

-- ============================================================================
-- TEXT SEARCH CONFIGURATION
-- ============================================================================

-- Set default text search configuration for future full-text search features
ALTER DATABASE rag_assistant SET default_text_search_config = 'pg_catalog.english';

-- ============================================================================
-- STATISTICS COLLECTION
-- ============================================================================

-- Increase statistics target for better query planning on vector columns
-- This helps PostgreSQL make better decisions about index usage
ALTER TABLE document_chunks ALTER COLUMN embedding SET STATISTICS 1000;

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Database configuration completed!';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Applied performance tuning for vector operations';
    RAISE NOTICE 'Configured autovacuum for optimal maintenance';
    RAISE NOTICE 'Set query timeouts and logging parameters';
    RAISE NOTICE '========================================';
END $$;
