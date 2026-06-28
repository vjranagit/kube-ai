"""Tests for policy/engine.py — saturation formula, swap override, unavailable metrics."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from controller.policy.engine import PolicyEngine
from controller.tests.conftest import make_cfg
from controller.types import ServingSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_snap(
    *,
    waiting: int = 0,
    running: int = 0,
    swapped: int = 0,
    kv_cache: float = 0.0,
    p95_ttft: float = 0.0,
    p50_ttft: float = 0.0,
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
        p50_ttft_sec=p50_ttft,
        queue_pressure=0.0,
        cache_pressure=0.0,
        latency_pressure=0.0,
        current_max_num_seqs=128,
        metrics_available=metrics_available,
    )


# ---------------------------------------------------------------------------
# metrics_available=False → saturation is 0.0
# ---------------------------------------------------------------------------


def test_saturation_zero_when_metrics_unavailable() -> None:
    policy = PolicyEngine(make_cfg())
    snap = make_snap(waiting=10, running=0, kv_cache=1.0, metrics_available=False)
    assert policy.saturation_score(snap) == pytest.approx(0.0)


def test_saturation_metrics_unavailable_ignores_queue() -> None:
    policy = PolicyEngine(make_cfg())
    snap = make_snap(waiting=100, running=0, metrics_available=False)
    assert policy.saturation_score(snap) == pytest.approx(0.0)


def test_saturation_metrics_unavailable_ignores_kv_cache() -> None:
    policy = PolicyEngine(make_cfg())
    snap = make_snap(kv_cache=1.0, metrics_available=False)
    assert policy.saturation_score(snap) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Queue pressure exactness
# ---------------------------------------------------------------------------


def test_queue_pressure_half_waiting() -> None:
    # waiting=5, running=5 → queue=5/10=0.5 → sat = 0.5*0.5 = 0.25
    policy = PolicyEngine(make_cfg(pressure_high=0.99, pressure_low=0.01))
    snap = make_snap(waiting=5, running=5)
    policy.saturation_score(snap)
    assert snap.queue_pressure == pytest.approx(0.5)


def test_queue_pressure_all_waiting() -> None:
    policy = PolicyEngine(make_cfg())
    snap = make_snap(waiting=10, running=0)
    policy.saturation_score(snap)
    assert snap.queue_pressure == pytest.approx(1.0)


def test_queue_pressure_none_waiting() -> None:
    policy = PolicyEngine(make_cfg())
    snap = make_snap(waiting=0, running=10)
    policy.saturation_score(snap)
    assert snap.queue_pressure == pytest.approx(0.0)


def test_queue_pressure_zero_requests_uses_denominator_one() -> None:
    # waiting=0, running=0 → total=max(1,0)=1 → queue=0/1=0.0 (no division by zero)
    policy = PolicyEngine(make_cfg())
    snap = make_snap(waiting=0, running=0)
    score = policy.saturation_score(snap)
    assert score >= 0.0  # no exception, result valid


# ---------------------------------------------------------------------------
# Cache pressure
# ---------------------------------------------------------------------------


def test_cache_pressure_propagated_from_kv_cache() -> None:
    policy = PolicyEngine(make_cfg())
    snap = make_snap(kv_cache=0.7)
    policy.saturation_score(snap)
    assert snap.cache_pressure == pytest.approx(0.7)


def test_cache_pressure_clamped_above_one() -> None:
    policy = PolicyEngine(make_cfg())
    snap = make_snap(kv_cache=1.5)
    policy.saturation_score(snap)
    assert snap.cache_pressure == pytest.approx(1.0)


def test_cache_pressure_clamped_below_zero() -> None:
    policy = PolicyEngine(make_cfg())
    snap = make_snap(kv_cache=-0.1)
    policy.saturation_score(snap)
    assert snap.cache_pressure == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Latency pressure
# ---------------------------------------------------------------------------


def test_latency_pressure_zero_when_ttft_below_slo() -> None:
    policy = PolicyEngine(make_cfg(ttft_slo_sec=2.0))
    snap = make_snap(p95_ttft=1.0)  # below SLO
    policy.saturation_score(snap)
    assert snap.latency_pressure == pytest.approx(0.0)


def test_latency_pressure_proportional_when_ttft_above_slo() -> None:
    # p95=4.0, slo=2.0 → (4-2)/2 = 1.0 → clamped to 1.0
    policy = PolicyEngine(make_cfg(ttft_slo_sec=2.0))
    snap = make_snap(p95_ttft=4.0)
    policy.saturation_score(snap)
    assert snap.latency_pressure == pytest.approx(1.0)


def test_latency_pressure_partial() -> None:
    # p95=3.0, slo=2.0 → (3-2)/2 = 0.5
    policy = PolicyEngine(make_cfg(ttft_slo_sec=2.0))
    snap = make_snap(p95_ttft=3.0)
    policy.saturation_score(snap)
    assert snap.latency_pressure == pytest.approx(0.5)


def test_latency_pressure_clamped_to_one() -> None:
    # Very high TTFT → latency_pressure stays ≤ 1
    policy = PolicyEngine(make_cfg(ttft_slo_sec=2.0))
    snap = make_snap(p95_ttft=100.0)
    policy.saturation_score(snap)
    assert snap.latency_pressure == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Composite formula exactness
# ---------------------------------------------------------------------------


def test_saturation_formula_weights_are_0_5_0_3_0_2() -> None:
    # Exact: 0.5*0.5 + 0.3*0.0 + 0.2*0.0 = 0.25
    policy = PolicyEngine(make_cfg())
    snap = make_snap(waiting=5, running=5, kv_cache=0.0, p95_ttft=0.0)
    score = policy.saturation_score(snap)
    assert score == pytest.approx(0.5 * 0.5)


def test_saturation_formula_cache_only() -> None:
    # queue=0, cache=0.6, latency=0 → 0.5*0 + 0.3*0.6 + 0.2*0 = 0.18
    policy = PolicyEngine(make_cfg())
    snap = make_snap(waiting=0, running=10, kv_cache=0.6, p95_ttft=0.0)
    score = policy.saturation_score(snap)
    assert score == pytest.approx(0.3 * 0.6)


def test_saturation_formula_latency_only() -> None:
    # queue=0, cache=0, latency=1.0 (p95=4.0, slo=2.0) → 0.2*1.0 = 0.2
    policy = PolicyEngine(make_cfg(ttft_slo_sec=2.0))
    snap = make_snap(waiting=0, running=10, kv_cache=0.0, p95_ttft=4.0)
    score = policy.saturation_score(snap)
    assert score == pytest.approx(0.2 * 1.0)


def test_saturation_typical_data_value() -> None:
    # waiting=3, running=5, kv_cache=0.42, p95=6.0, slo=2.0
    # queue=3/8=0.375, cache=0.42, latency=clamp(2,0,1)=1.0
    # sat = 0.5*0.375 + 0.3*0.42 + 0.2*1.0 = 0.1875 + 0.126 + 0.2 = 0.5135
    policy = PolicyEngine(make_cfg(ttft_slo_sec=2.0))
    snap = make_snap(waiting=3, running=5, kv_cache=0.42, p95_ttft=6.0)
    score = policy.saturation_score(snap)
    assert score == pytest.approx(0.5135)


def test_saturation_result_clamped_at_one() -> None:
    policy = PolicyEngine(make_cfg())
    snap = make_snap(waiting=100, running=0, kv_cache=1.0, p95_ttft=100.0)
    score = policy.saturation_score(snap)
    assert score <= 1.0


def test_saturation_result_never_below_zero() -> None:
    policy = PolicyEngine(make_cfg())
    snap = make_snap(waiting=0, running=0, kv_cache=0.0, p95_ttft=0.0)
    score = policy.saturation_score(snap)
    assert score >= 0.0


# ---------------------------------------------------------------------------
# Swap override — swapped > 0 floors saturation above pressure_high
# ---------------------------------------------------------------------------


def test_swap_override_floors_saturation_above_pressure_high() -> None:
    cfg = make_cfg(pressure_high=0.75)
    policy = PolicyEngine(cfg)
    # Artificially low composite score, but swapped=1 triggers override
    snap = make_snap(waiting=0, running=10, swapped=1, kv_cache=0.0, p95_ttft=0.0)
    score = policy.saturation_score(snap)
    assert score >= cfg.pressure_high + 0.01


def test_swap_override_exactly_above_pressure_high_plus_epsilon() -> None:
    cfg = make_cfg(pressure_high=0.75)
    policy = PolicyEngine(cfg)
    snap = make_snap(waiting=0, running=10, swapped=2, kv_cache=0.0, p95_ttft=0.0)
    score = policy.saturation_score(snap)
    assert score == pytest.approx(0.76)


def test_swap_override_does_not_lower_high_composite_score() -> None:
    # If the composite score is already > pressure_high + 0.01, swap doesn't lower it
    cfg = make_cfg(pressure_high=0.1)
    policy = PolicyEngine(cfg)
    # Compose score ≈ 0.5 * 1.0 = 0.5 > 0.11
    snap = make_snap(waiting=10, running=0, swapped=1, kv_cache=0.0, p95_ttft=0.0)
    score = policy.saturation_score(snap)
    assert score >= 0.5


def test_swap_zero_does_not_trigger_override() -> None:
    cfg = make_cfg(pressure_high=0.75)
    policy = PolicyEngine(cfg)
    snap = make_snap(waiting=0, running=10, swapped=0, kv_cache=0.0, p95_ttft=0.0)
    score = policy.saturation_score(snap)
    assert score < cfg.pressure_high  # no override — should be 0


# ---------------------------------------------------------------------------
# Sub-scores written back to snapshot fields
# ---------------------------------------------------------------------------


def test_snapshot_queue_pressure_updated_in_place() -> None:
    policy = PolicyEngine(make_cfg())
    snap = make_snap(waiting=3, running=7)
    policy.saturation_score(snap)
    assert snap.queue_pressure == pytest.approx(3 / 10)


def test_snapshot_cache_pressure_updated_in_place() -> None:
    policy = PolicyEngine(make_cfg())
    snap = make_snap(kv_cache=0.55)
    policy.saturation_score(snap)
    assert snap.cache_pressure == pytest.approx(0.55)


def test_snapshot_latency_pressure_updated_in_place() -> None:
    policy = PolicyEngine(make_cfg(ttft_slo_sec=2.0))
    snap = make_snap(p95_ttft=3.0)
    policy.saturation_score(snap)
    assert snap.latency_pressure == pytest.approx(0.5)
