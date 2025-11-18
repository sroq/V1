/**
 * OpenTelemetry Instrumentation for Next.js Application
 *
 * This file is automatically loaded by Next.js before the application starts.
 * It sets up OpenTelemetry tracing and metrics for the RAG assistant.
 *
 * Architecture:
 * - TracerProvider: Creates and manages spans (traces)
 * - MeterProvider: Creates and manages metrics (costs, tokens)
 * - OTLP Exporters: Send traces and metrics to Jaeger (http://localhost:4318)
 * - Auto-instrumentation: PostgreSQL queries are auto-traced
 *
 * Next.js automatically calls the `register()` function on startup.
 */

/**
 * register() is called automatically by Next.js when the application starts.
 *
 * We're using a minimal setup that works with Next.js 15.
 */
export async function register() {
  // Only initialize in server-side (Node.js runtime)
  if (process.env.NEXT_RUNTIME === 'nodejs') {
    try {
      // Dynamic import to avoid bundling issues
      const { NodeSDK } = await import('@opentelemetry/sdk-node');
      const { OTLPTraceExporter } = await import('@opentelemetry/exporter-trace-otlp-http');
      const { OTLPMetricExporter } = await import('@opentelemetry/exporter-metrics-otlp-http');
      const { PeriodicExportingMetricReader } = await import('@opentelemetry/sdk-metrics');
      const { getNodeAutoInstrumentations } = await import('@opentelemetry/auto-instrumentations-node');

      const JAEGER_ENDPOINT = process.env.OTEL_EXPORTER_OTLP_ENDPOINT || 'http://localhost:4318';
      const SERVICE_NAME = process.env.OTEL_SERVICE_NAME || 'rag-assistant';

      // Configure OTLP trace exporter
      const traceExporter = new OTLPTraceExporter({
        url: `${JAEGER_ENDPOINT}/v1/traces`,
      });

      // Configure OTLP metrics exporter
      const metricExporter = new OTLPMetricExporter({
        url: `${JAEGER_ENDPOINT}/v1/metrics`,
      });

      // Configure metric reader (export every 60 seconds)
      const metricReader = new PeriodicExportingMetricReader({
        exporter: metricExporter,
        exportIntervalMillis: 60000, // 60 seconds
      });

      // Configure auto-instrumentation
      const instrumentations = getNodeAutoInstrumentations({
        // HTTP instrumentation
        '@opentelemetry/instrumentation-http': {
          enabled: true,
        },
        // PostgreSQL instrumentation (pgvector queries!)
        '@opentelemetry/instrumentation-pg': {
          enabled: true,
          enhancedDatabaseReporting: true,
        },
        // Disable noisy instrumentations
        '@opentelemetry/instrumentation-dns': { enabled: false },
        '@opentelemetry/instrumentation-net': { enabled: false },
        '@opentelemetry/instrumentation-fs': { enabled: false },
      });

      // Initialize SDK with both traces and metrics
      const sdk = new NodeSDK({
        serviceName: SERVICE_NAME,
        traceExporter,
        metricReader,
        instrumentations,
      });

      await sdk.start();

      console.log('[OpenTelemetry] Initialized successfully');
      console.log(`[OpenTelemetry] Service: ${SERVICE_NAME}`);
      console.log(`[OpenTelemetry] Exporting to: ${JAEGER_ENDPOINT}`);
      console.log(`[OpenTelemetry] Jaeger UI: http://localhost:16686`);
      console.log(`[OpenTelemetry] Metrics export interval: 60s`);

      // Graceful shutdown
      process.on('SIGTERM', async () => {
        await sdk.shutdown();
        console.log('[OpenTelemetry] Shutdown complete');
      });
    } catch (error) {
      console.error('[OpenTelemetry] Failed to initialize:', error);
    }
  }
}
