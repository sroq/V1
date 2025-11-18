-- Migration 003: Rollback API Metrics Tables
-- Description: Removes custom monitoring tables (migrating to OpenTelemetry)
-- Author: Claude Code
-- Date: 2025-11-15

-- Drop views first (they depend on the table)
DROP VIEW IF EXISTS v_session_costs;
DROP VIEW IF EXISTS v_latency_percentiles;
DROP VIEW IF EXISTS v_cost_summary;

-- Drop table
DROP TABLE IF EXISTS api_metrics CASCADE;

-- Log completion
SELECT 'API metrics tables successfully removed' AS status;
