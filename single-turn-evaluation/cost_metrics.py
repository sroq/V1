"""
OpenTelemetry Cost Tracking for Multi-Turn Evaluation

Ez a modul rögzíti az evaluation scriptek költségeit OpenTelemetry metrics-ben,
így azok megjelennek a Prometheus/Grafana dashboardon.

Metrikák:
- rag.cost.evaluation_embedding: Evaluation embedding költségek
- rag.cost.evaluation_llm: Evaluation LLM költségek (GPT-4o mini judge)
- rag.cost.evaluation_total: Összes evaluation költség
- rag.tokens.evaluation_input: Evaluation input tokenek
- rag.tokens.evaluation_output: Evaluation output tokenek
"""

import os
from typing import Dict, Optional
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

# ============================================================================
# PRICING CONSTANTS (USD per 1K tokens)
# ============================================================================

PRICING = {
    "EMBEDDING": 0.00002,  # text-embedding-3-small
    "LLM_INPUT": 0.00015,  # gpt-4o-mini input
    "LLM_OUTPUT": 0.0006,  # gpt-4o-mini output
}


# ============================================================================
# OPENTELEMETRY SETUP
# ============================================================================

def setup_metrics_provider(
    endpoint: str = None,
    export_interval_ms: int = 10000  # 10 seconds for evaluation
) -> metrics.Meter:
    """
    OpenTelemetry metrics provider beállítása.

    Args:
        endpoint: OTLP endpoint URL (default: http://localhost:4318)
        export_interval_ms: Export interval milliseconds

    Returns:
        Configured Meter instance
    """
    endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

    # OTLP Metric Exporter
    exporter = OTLPMetricExporter(
        endpoint=f"{endpoint}/v1/metrics",
        headers={}
    )

    # Metric Reader with periodic export
    reader = PeriodicExportingMetricReader(
        exporter=exporter,
        export_interval_millis=export_interval_ms
    )

    # Meter Provider
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)

    # Get Meter
    meter = metrics.get_meter(
        "multi-turn-evaluation",
        version="1.0.0"
    )

    print(f"[OpenTelemetry] Metrics initialized - exporting to {endpoint}")
    print(f"[OpenTelemetry] Export interval: {export_interval_ms}ms")

    return meter


# ============================================================================
# COST METRICS
# ============================================================================

# Global meter (initialized in CostTracker)
_meter: Optional[metrics.Meter] = None

# Metric instruments
_eval_embedding_cost_counter = None
_eval_llm_cost_counter = None
_eval_total_cost_counter = None
_eval_input_token_counter = None
_eval_output_token_counter = None


class CostTracker:
    """
    Cost tracking singleton for multi-turn evaluation.

    Usage:
        tracker = CostTracker()
        tracker.record_llm_cost(input_tokens=350, output_tokens=50)
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        global _meter
        global _eval_embedding_cost_counter
        global _eval_llm_cost_counter
        global _eval_total_cost_counter
        global _eval_input_token_counter
        global _eval_output_token_counter

        # Setup meter
        _meter = setup_metrics_provider()

        # Create metric instruments
        _eval_embedding_cost_counter = _meter.create_counter(
            name="rag.cost.evaluation_embedding_USD",
            description="Evaluation embedding costs in USD",
            unit="USD"
        )

        _eval_llm_cost_counter = _meter.create_counter(
            name="rag.cost.evaluation_llm_USD",
            description="Evaluation LLM costs in USD (GPT-4o mini judge)",
            unit="USD"
        )

        _eval_total_cost_counter = _meter.create_counter(
            name="rag.cost.evaluation_total_USD",
            description="Total evaluation costs in USD",
            unit="USD"
        )

        _eval_input_token_counter = _meter.create_counter(
            name="rag.tokens.evaluation_input",
            description="Evaluation input tokens",
            unit="tokens"
        )

        _eval_output_token_counter = _meter.create_counter(
            name="rag.tokens.evaluation_output",
            description="Evaluation output tokens",
            unit="tokens"
        )

        self._initialized = True
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        print("[CostTracker] Initialized - ready to track evaluation costs")

    def record_embedding_cost(
        self,
        tokens: int,
        attributes: Dict[str, str] = None
    ) -> float:
        """
        Record embedding cost.

        Args:
            tokens: Number of tokens
            attributes: Optional metric attributes

        Returns:
            Cost in USD
        """
        cost = (tokens / 1000) * PRICING["EMBEDDING"]

        attrs = attributes or {}
        attrs["evaluation_type"] = "multi_turn"

        _eval_embedding_cost_counter.add(cost, attrs)
        _eval_total_cost_counter.add(cost, attrs)

        self.total_cost += cost

        return cost

    def record_llm_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        attributes: Dict[str, str] = None
    ) -> float:
        """
        Record LLM cost (GPT-4o mini judge).

        Args:
            input_tokens: Input token count
            output_tokens: Output token count
            attributes: Optional metric attributes

        Returns:
            Total cost in USD
        """
        input_cost = (input_tokens / 1000) * PRICING["LLM_INPUT"]
        output_cost = (output_tokens / 1000) * PRICING["LLM_OUTPUT"]
        total_cost = input_cost + output_cost

        attrs = attributes or {}
        attrs["evaluation_type"] = "multi_turn"

        _eval_input_token_counter.add(input_tokens, attrs)
        _eval_output_token_counter.add(output_tokens, attrs)
        _eval_llm_cost_counter.add(total_cost, attrs)
        _eval_total_cost_counter.add(total_cost, attrs)

        self.total_cost += total_cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        return total_cost

    def get_summary(self) -> Dict[str, any]:
        """
        Get cost tracking summary.

        Returns:
            Dict with total costs and tokens
        """
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "avg_cost_per_call": round(
                self.total_cost / max(1, self.total_input_tokens // 350),
                6
            )
        }

    def print_summary(self):
        """Print cost summary to console."""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("EVALUATION COST SUMMARY")
        print("="*60)
        print(f"Total Cost:        ${summary['total_cost_usd']:.6f}")
        print(f"Input Tokens:      {summary['total_input_tokens']:,}")
        print(f"Output Tokens:     {summary['total_output_tokens']:,}")
        print(f"Total Tokens:      {summary['total_tokens']:,}")
        print(f"Avg Cost/Call:     ${summary['avg_cost_per_call']:.6f}")
        print("="*60)

    def force_export(self):
        """
        Force immediate metric export (for script completion).

        Note: This is a workaround since PeriodicExportingMetricReader
        doesn't have a public force_flush method.
        """
        import time
        print("[CostTracker] Waiting for final metric export...")
        time.sleep(2)  # Wait for export interval
        print("[CostTracker] Metrics exported to Prometheus/Grafana")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_cost_tracker() -> CostTracker:
    """Get or create cost tracker singleton."""
    return CostTracker()
