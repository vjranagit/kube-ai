"""Pressure / batch-load tests.

These exercise the decision path under escalating load *tiers* — the same shape the real
concurrent load generator (scripts/load-real.py) produces against a live vLLM endpoint —
but as pure, deterministic, CI-safe unit tests (no model, no cluster, no GPU).

They assert:
  * saturation rises monotonically as queue/cache/latency pressure climb,
  * the tuner scales OUT (and raises max_num_seqs) once saturation crosses pressure_high,
  * the tuner scales IN once saturation falls below pressure_low,
  * bounds are never violated across a full sweep,
  * the swapped-request hard override forces scale-out.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from controller.policy.engine import PolicyEngine
from controller.tests.conftest import make_cfg
from controller.tuner.aimd import AimdTuner
from controller.types import ServingSnapshot


def _snap(
    *,
    waiting: int = 0,
    running: int = 0,
    swapped: int = 0,
    kv_cache: float = 0.0,
    p95_ttft: float = 0.0,
    metrics_available: bool = True,
) -> ServingSnapshot:
    return ServingSnapshot(
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        desired_replicas=2,
        ready_replicas=2,
        available_replicas=2,
        requests_waiting=waiting,
        requests_running=running,
        requests_swapped=swapped,
        kv_cache_usage_perc=kv_cache,
        p95_ttft_sec=p95_ttft,
        p50_ttft_sec=p95_ttft / 2.0,
        queue_pressure=0.0,
        cache_pressure=0.0,
        latency_pressure=0.0,
        current_max_num_seqs=256,
        metrics_available=metrics_available,
    )


# Load tiers modeled on scripts/load-real.py concurrency levels: idle -> light -> heavy -> saturated.
# Each tier raises queue depth, KV-cache usage, and TTFT together, like a real backend under
# increasing concurrency.
TIERS = [
    {"name": "idle", "waiting": 0, "running": 1, "kv_cache": 0.02, "p95_ttft": 0.1},
    {"name": "light", "waiting": 4, "running": 6, "kv_cache": 0.25, "p95_ttft": 1.0},
    {"name": "heavy", "waiting": 30, "running": 8, "kv_cache": 0.70, "p95_ttft": 6.0},
    {"name": "saturated", "waiting": 120, "running": 8, "kv_cache": 0.95, "p95_ttft": 14.0},
]


def test_saturation_is_monotonic_across_load_tiers() -> None:
    policy = PolicyEngine(make_cfg(ttft_slo_sec=2.0))
    scores = [policy.saturation_score(_snap(**{k: v for k, v in t.items() if k != "name"})) for t in TIERS]
    for lo, hi in zip(scores, scores[1:]):
        assert hi > lo, f"saturation not increasing across tiers: {scores}"
    assert scores[0] < 0.2  # idle is calm
    assert scores[-1] >= 0.8  # saturated tier is hot


def test_idle_tier_triggers_scale_in() -> None:
    cfg = make_cfg(ttft_slo_sec=2.0)
    policy, tuner = PolicyEngine(cfg), AimdTuner(cfg)
    sat = policy.saturation_score(_snap(waiting=0, running=1, kv_cache=0.02, p95_ttft=0.1))
    assert sat <= cfg.pressure_low
    assert tuner.next_replicas(4, sat) == max(cfg.min_replicas, 4 // 2)  # multiplicative decrease


def test_saturated_tier_triggers_scale_out_and_raises_params() -> None:
    cfg = make_cfg(ttft_slo_sec=2.0)
    policy, tuner = PolicyEngine(cfg), AimdTuner(cfg)
    sat = policy.saturation_score(_snap(waiting=120, running=8, kv_cache=0.95, p95_ttft=14.0))
    assert sat >= cfg.pressure_high
    cur = max(cfg.min_replicas, 1)
    assert tuner.next_replicas(cur, sat) == min(cfg.max_replicas, cur + 1)  # additive increase
    assert tuner.next_max_num_seqs(cfg.min_max_num_seqs, sat) > cfg.min_max_num_seqs


def test_swapped_override_forces_scale_out_even_when_queue_calm() -> None:
    cfg = make_cfg()
    policy, tuner = PolicyEngine(cfg), AimdTuner(cfg)
    # No queue/cache/latency pressure, but a swapped request = KV preemption.
    sat = policy.saturation_score(_snap(waiting=0, running=2, kv_cache=0.0, p95_ttft=0.0, swapped=1))
    assert sat > cfg.pressure_high
    assert tuner.next_replicas(2, sat) == min(cfg.max_replicas, 3)


def test_bounds_never_violated_under_full_pressure_sweep() -> None:
    cfg = make_cfg()
    policy, tuner = PolicyEngine(cfg), AimdTuner(cfg)
    current = cfg.min_replicas
    seqs = cfg.min_max_num_seqs
    # Sweep many tiers up and down repeatedly; replicas/seqs must always stay in bounds.
    for waiting in [0, 5, 50, 200, 50, 5, 0] * 5:
        sat = policy.saturation_score(
            _snap(waiting=waiting, running=8, kv_cache=min(0.99, waiting / 200), p95_ttft=waiting / 10)
        )
        current = tuner.next_replicas(current, sat)
        seqs = tuner.next_max_num_seqs(seqs, sat)
        assert cfg.min_replicas <= current <= cfg.max_replicas
        assert cfg.min_max_num_seqs <= seqs <= cfg.max_max_num_seqs


def test_metrics_unavailable_is_calm_no_scale_out() -> None:
    cfg = make_cfg()
    policy, tuner = PolicyEngine(cfg), AimdTuner(cfg)
    sat = policy.saturation_score(_snap(waiting=999, running=0, kv_cache=1.0, metrics_available=False))
    assert sat == pytest.approx(0.0)
    # With saturation 0 (<= pressure_low) the loop must not scale out into a broken backend.
    assert tuner.next_replicas(3, sat) <= 3
