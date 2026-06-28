"""PolicyEngine — pure, stateless saturation scoring.

Saturation formula (composite [0..1]):
    saturation = 0.50 * queue_pressure
               + 0.30 * cache_pressure
               + 0.20 * latency_pressure

where:
    queue_pressure   = waiting / max(1, waiting + running)
    cache_pressure   = vllm:kv_cache_usage_perc            [0..1]
    latency_pressure = clamp((p95_ttft - TTFT_SLO) / TTFT_SLO, 0, 1)

Hard override: if requests_swapped > 0, saturation is floored just above pressure_high.
This reflects the Chiron paper's insight that swapped requests indicate KV preemption —
a more severe signal than the composite score alone.

If metrics_available is False, returns 0.0 (no action into a broken cluster).

NOTE: This engine also mutates the snapshot's sub-score fields so they appear in logs
and API responses without a second computation.
"""

from __future__ import annotations

from controller.config import ControllerConfig
from controller.types import ServingSnapshot


class PolicyEngine:
    def __init__(self, cfg: ControllerConfig) -> None:
        self.cfg = cfg

    def saturation_score(self, snap: ServingSnapshot) -> float:
        """Compute saturation score and update snap sub-scores in-place.

        Returns a float in [0..1] (or just above pressure_high on swap override).
        Returns 0.0 if metrics are unavailable.
        """
        if not snap.metrics_available:
            return 0.0

        # Sub-scores
        total_requests = max(1, snap.requests_waiting + snap.requests_running)
        queue_pressure = snap.requests_waiting / total_requests

        cache_pressure = max(0.0, min(1.0, snap.kv_cache_usage_perc))

        slo = max(1e-6, self.cfg.ttft_slo_sec)
        latency_pressure = max(0.0, min(1.0, (snap.p95_ttft_sec - slo) / slo))

        # Update snapshot fields so callers can log/export them without recomputing
        snap.queue_pressure = queue_pressure
        snap.cache_pressure = cache_pressure
        snap.latency_pressure = latency_pressure

        score = (
            0.50 * queue_pressure
            + 0.30 * cache_pressure
            + 0.20 * latency_pressure
        )

        # Hard override: any swapped requests → floor above pressure_high
        if snap.requests_swapped > 0:
            score = max(score, self.cfg.pressure_high + 0.01)

        return min(1.0, score)
