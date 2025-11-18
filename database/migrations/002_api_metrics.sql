-- Migration 002: API Metrics & Monitoring Tables
-- Description: Creates tables and views for comprehensive API monitoring
-- Author: Claude Code
-- Date: 2025-11-15

-- ============================================================================
-- 1. API METRICS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS api_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Relationships
  session_id VARCHAR(36) REFERENCES chat_sessions(id) ON DELETE SET NULL,
  message_id VARCHAR(36) REFERENCES chat_messages(id) ON DELETE SET NULL,

  -- Request metadata
  endpoint VARCHAR(100) NOT NULL DEFAULT '/api/chat',
  method VARCHAR(10) NOT NULL DEFAULT 'POST',
  status_code INTEGER,

  -- Latency breakdown (milliseconds)
  embedding_latency_ms INTEGER,        -- Embedding generation time
  vector_search_latency_ms INTEGER,    -- pgvector cosine similarity
  reranking_latency_ms INTEGER,        -- 15 parallel GPT-4o mini calls
  llm_streaming_latency_ms INTEGER,    -- Main chat completion
  ttft_ms INTEGER,                     -- Time To First Token
  total_latency_ms INTEGER NOT NULL,   -- End-to-end request time

  -- Cost breakdown (USD, 6 decimal precision)
  embedding_cost DECIMAL(10,6) DEFAULT 0,      -- text-embedding-3-small
  reranking_cost DECIMAL(10,6) DEFAULT 0,      -- 15× GPT-4o mini scoring
  llm_cost DECIMAL(10,6) NOT NULL,             -- Main chat completion
  total_cost DECIMAL(10,6) NOT NULL,           -- Sum of all costs

  -- Token usage (from OpenAI API response)
  prompt_tokens INTEGER NOT NULL,
  completion_tokens INTEGER NOT NULL,
  total_tokens INTEGER NOT NULL,

  -- RAG context metadata
  chunks_retrieved INTEGER DEFAULT 0,   -- Initial vector search (15)
  chunks_reranked INTEGER DEFAULT 0,    -- Reranked count (15)
  chunks_used INTEGER DEFAULT 0,        -- Final top-K (5)
  context_token_count INTEGER,          -- Estimated tokens in RAG context

  -- Error tracking
  error_occurred BOOLEAN DEFAULT FALSE,
  error_type VARCHAR(100),
  error_message TEXT,

  -- Timestamps
  created_at TIMESTAMP DEFAULT NOW(),

  -- Additional metadata (JSONB for flexibility)
  metadata JSONB
);

-- ============================================================================
-- 2. INDEXES
-- ============================================================================

-- Primary lookup indexes
CREATE INDEX idx_metrics_session ON api_metrics(session_id);
CREATE INDEX idx_metrics_message ON api_metrics(message_id);
CREATE INDEX idx_metrics_created_at ON api_metrics(created_at DESC);
CREATE INDEX idx_metrics_endpoint ON api_metrics(endpoint);

-- Error tracking index (filtered)
CREATE INDEX idx_metrics_error ON api_metrics(error_occurred) WHERE error_occurred = TRUE;

-- Composite index for time-range queries
CREATE INDEX idx_metrics_time_endpoint ON api_metrics(created_at DESC, endpoint);

-- GIN index for metadata queries
CREATE INDEX idx_metrics_metadata ON api_metrics USING GIN(metadata);

-- ============================================================================
-- 3. AGGREGATED VIEWS
-- ============================================================================

-- View 1: Daily cost summary
CREATE OR REPLACE VIEW v_cost_summary AS
SELECT
  DATE(created_at) as date,
  COUNT(*) as total_requests,
  SUM(total_cost) as total_cost_usd,
  SUM(embedding_cost) as embedding_cost_usd,
  SUM(reranking_cost) as reranking_cost_usd,
  SUM(llm_cost) as llm_cost_usd,
  SUM(total_tokens) as total_tokens,
  SUM(prompt_tokens) as prompt_tokens,
  SUM(completion_tokens) as completion_tokens,
  AVG(total_latency_ms) as avg_latency_ms,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_latency_ms) as p95_latency_ms,
  PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY total_latency_ms) as p99_latency_ms,
  SUM(CASE WHEN error_occurred THEN 1 ELSE 0 END) as error_count,
  (SUM(CASE WHEN error_occurred THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT * 100) as error_rate_percent
FROM api_metrics
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- View 2: Latency percentiles (daily)
CREATE OR REPLACE VIEW v_latency_percentiles AS
SELECT
  DATE(created_at) as date,
  COUNT(*) as sample_count,
  -- Total latency
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_latency_ms) as p50_total_ms,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_latency_ms) as p95_total_ms,
  PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY total_latency_ms) as p99_total_ms,
  AVG(total_latency_ms) as avg_total_ms,
  -- TTFT (Time To First Token)
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY ttft_ms) as p50_ttft_ms,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ttft_ms) as p95_ttft_ms,
  AVG(ttft_ms) as avg_ttft_ms,
  -- Embedding latency
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY embedding_latency_ms) as p50_embedding_ms,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY embedding_latency_ms) as p95_embedding_ms,
  AVG(embedding_latency_ms) as avg_embedding_ms,
  -- Reranking latency
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY reranking_latency_ms) as p50_reranking_ms,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY reranking_latency_ms) as p95_reranking_ms,
  AVG(reranking_latency_ms) as avg_reranking_ms,
  -- LLM streaming latency
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY llm_streaming_latency_ms) as p50_llm_ms,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY llm_streaming_latency_ms) as p95_llm_ms,
  AVG(llm_streaming_latency_ms) as avg_llm_ms
FROM api_metrics
WHERE total_latency_ms IS NOT NULL
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- View 3: Session-level costs
CREATE OR REPLACE VIEW v_session_costs AS
SELECT
  am.session_id,
  cs.title,
  cs.started_at,
  cs.last_activity_at,
  COUNT(am.id) as request_count,
  SUM(am.total_cost) as total_cost_usd,
  SUM(am.total_tokens) as total_tokens,
  SUM(am.prompt_tokens) as prompt_tokens,
  SUM(am.completion_tokens) as completion_tokens,
  AVG(am.total_latency_ms) as avg_latency_ms,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY am.total_latency_ms) as p95_latency_ms,
  AVG(am.ttft_ms) as avg_ttft_ms,
  SUM(CASE WHEN am.error_occurred THEN 1 ELSE 0 END) as error_count,
  -- Cost breakdown
  SUM(am.embedding_cost) as embedding_cost_usd,
  SUM(am.reranking_cost) as reranking_cost_usd,
  SUM(am.llm_cost) as llm_cost_usd,
  -- RAG metrics
  AVG(am.chunks_used) as avg_chunks_used
FROM api_metrics am
JOIN chat_sessions cs ON am.session_id = cs.id
GROUP BY am.session_id, cs.title, cs.started_at, cs.last_activity_at
ORDER BY total_cost_usd DESC;

-- ============================================================================
-- 4. COMMENTS
-- ============================================================================

COMMENT ON TABLE api_metrics IS 'Comprehensive API request monitoring metrics';
COMMENT ON COLUMN api_metrics.ttft_ms IS 'Time To First Token - measures streaming responsiveness';
COMMENT ON COLUMN api_metrics.embedding_cost IS 'Cost of text-embedding-3-small API call';
COMMENT ON COLUMN api_metrics.reranking_cost IS 'Cost of 15 parallel GPT-4o mini reranking calls';
COMMENT ON COLUMN api_metrics.llm_cost IS 'Cost of main chat completion (GPT-4o mini)';
COMMENT ON COLUMN api_metrics.context_token_count IS 'Estimated token count in assembled RAG context';

COMMENT ON VIEW v_cost_summary IS 'Daily aggregated cost and usage metrics';
COMMENT ON VIEW v_latency_percentiles IS 'Daily latency distributions (P50, P95, P99)';
COMMENT ON VIEW v_session_costs IS 'Per-session cost and performance metrics';

-- ============================================================================
-- 5. GRANTS (adjust as needed for your security model)
-- ============================================================================

-- Grant read access to views (adjust role as needed)
-- GRANT SELECT ON v_cost_summary TO dashboard_user;
-- GRANT SELECT ON v_latency_percentiles TO dashboard_user;
-- GRANT SELECT ON v_session_costs TO dashboard_user;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

-- To verify:
-- SELECT COUNT(*) FROM api_metrics;
-- SELECT * FROM v_cost_summary LIMIT 5;
-- SELECT * FROM v_latency_percentiles LIMIT 5;
-- SELECT * FROM v_session_costs LIMIT 10;
