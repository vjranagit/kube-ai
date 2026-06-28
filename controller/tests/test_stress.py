"""2 000-iteration end-to-end stress tests: collector mock → policy → tuner → actuator."""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from controller.actuator.k8s import K8sActuator
from controller.policy.engine import PolicyEngine
from controller.tests.conftest import make_cfg
from controller.tuner.aimd import AimdTuner
from controller.types import PolicyDecision, ServingSnapshot

ITERATIONS = 2000
SEED = 42


# ---------------------------------------------------------------------------
# Snapshot factory
# ---------------------------------------------------------------------------


def make_snap(
    *,
    waiting: int = 0,
    running: int = 0,
    swapped: int = 0,
    kv_cache: float = 0.0,
    p95_ttft: float = 0.0,
    metrics_available: bool = True,
) -> ServingSnapshot:
    return ServingSnapshot(
        timestamp=datetime.now(timezone.utc),
        desired_replicas=2,
        ready_replicas=2,
        available_replicas=2,
        requests_waiting=waiting,
        requests_running=running,
        requests_swapped=swapped,
        kv_cache_usage_perc=kv_cache,
        p95_ttft_sec=p95_ttft,
        p50_ttft_sec=0.0,
        queue_pressure=0.0,
        cache_pressure=0.0,
        latency_pressure=0.0,
        current_max_num_seqs=128,
        metrics_available=metrics_available,
    )


def expire_cooldowns(actuator: K8sActuator) -> None:
    past = datetime.now(timezone.utc) - timedelta(hours=24)
    actuator.last_replica_apply = past
    actuator.last_param_apply = past


# ---------------------------------------------------------------------------
# Full loop invariants
# ---------------------------------------------------------------------------


def test_stress_replicas_always_within_bounds() -> None:
    """After 2000 random-saturation steps, replicas always stays in [min, max]."""
    cfg = make_cfg(
        min_replicas=1,
        max_replicas=8,
        min_max_num_seqs=128,
        max_max_num_seqs=2048,
        cooldown_sec=0,
        param_cooldown_sec=0,
        dry_run=True,
        tune_mode="both",
    )
    policy = PolicyEngine(cfg)
    tuner = AimdTuner(cfg)
    actuator = K8sActuator(cfg)
    rng = random.Random(SEED)

    for i in range(ITERATIONS):
        waiting = rng.randint(0, 20)
        running = rng.randint(0, 20)
        kv_cache = rng.random()
        p95 = rng.uniform(0.0, 6.0)
        snap = make_snap(waiting=waiting, running=running, kv_cache=kv_cache, p95_ttft=p95)

        sat = policy.saturation_score(snap)
        target_replicas = tuner.next_replicas(actuator.state.current_replicas, sat)
        target_seqs = tuner.next_max_num_seqs(actuator.state.current_max_num_seqs, sat)
        decision = PolicyDecision(
            target_replicas=target_replicas,
            target_max_num_seqs=target_seqs,
            saturation=sat,
            reason="test",
        )

        expire_cooldowns(actuator)
        actuator.apply(decision)

        assert cfg.min_replicas <= actuator.state.current_replicas <= cfg.max_replicas, (
            f"iter={i} replicas={actuator.state.current_replicas} out of "
            f"[{cfg.min_replicas}, {cfg.max_replicas}]"
        )


def test_stress_max_num_seqs_always_within_bounds() -> None:
    cfg = make_cfg(
        min_replicas=1,
        max_replicas=8,
        min_max_num_seqs=128,
        max_max_num_seqs=2048,
        cooldown_sec=0,
        param_cooldown_sec=0,
        dry_run=True,
        tune_mode="both",
    )
    policy = PolicyEngine(cfg)
    tuner = AimdTuner(cfg)
    actuator = K8sActuator(cfg)
    rng = random.Random(SEED)

    for i in range(ITERATIONS):
        waiting = rng.randint(0, 20)
        running = rng.randint(0, 20)
        snap = make_snap(waiting=waiting, running=running, kv_cache=rng.random())

        sat = policy.saturation_score(snap)
        target_seqs = tuner.next_max_num_seqs(actuator.state.current_max_num_seqs, sat)
        decision = PolicyDecision(
            target_replicas=tuner.next_replicas(actuator.state.current_replicas, sat),
            target_max_num_seqs=target_seqs,
            saturation=sat,
            reason="test",
        )

        expire_cooldowns(actuator)
        actuator.apply(decision)

        assert cfg.min_max_num_seqs <= actuator.state.current_max_num_seqs <= cfg.max_max_num_seqs, (
            f"iter={i} seqs={actuator.state.current_max_num_seqs} out of "
            f"[{cfg.min_max_num_seqs}, {cfg.max_max_num_seqs}]"
        )


def test_stress_no_exceptions_under_random_pressure() -> None:
    """End-to-end loop must not raise any exception over 2000 iterations."""
    cfg = make_cfg(cooldown_sec=0, param_cooldown_sec=0, dry_run=True, tune_mode="both")
    policy = PolicyEngine(cfg)
    tuner = AimdTuner(cfg)
    actuator = K8sActuator(cfg)
    rng = random.Random(SEED)

    for _ in range(ITERATIONS):
        snap = make_snap(
            waiting=rng.randint(0, 50),
            running=rng.randint(0, 50),
            swapped=rng.randint(0, 5),
            kv_cache=rng.random(),
            p95_ttft=rng.uniform(0, 10),
        )
        sat = policy.saturation_score(snap)
        decision = PolicyDecision(
            target_replicas=tuner.next_replicas(actuator.state.current_replicas, sat),
            target_max_num_seqs=tuner.next_max_num_seqs(actuator.state.current_max_num_seqs, sat),
            saturation=sat,
            reason="test",
        )
        expire_cooldowns(actuator)
        actuator.apply(decision)


def test_stress_saturation_always_in_range() -> None:
    """Saturation score is always in [0, 1] regardless of input signals."""
    cfg = make_cfg(pressure_high=0.75, pressure_low=0.35, ttft_slo_sec=2.0)
    policy = PolicyEngine(cfg)
    rng = random.Random(SEED)

    for _ in range(ITERATIONS):
        snap = make_snap(
            waiting=rng.randint(0, 100),
            running=rng.randint(0, 100),
            swapped=rng.randint(0, 10),
            kv_cache=rng.uniform(0, 2),  # intentionally outside [0,1] to test clamping
            p95_ttft=rng.uniform(0, 20),
        )
        sat = policy.saturation_score(snap)
        assert 0.0 <= sat <= 1.0, f"saturation={sat} outside [0, 1]"


def test_stress_metrics_unavailable_keeps_saturation_zero() -> None:
    """When metrics_available=False, saturation is always 0 no matter the signals."""
    cfg = make_cfg()
    policy = PolicyEngine(cfg)
    rng = random.Random(SEED)

    for _ in range(ITERATIONS):
        snap = make_snap(
            waiting=rng.randint(0, 100),
            running=rng.randint(0, 100),
            swapped=rng.randint(0, 10),
            kv_cache=rng.random(),
            p95_ttft=rng.uniform(0, 10),
            metrics_available=False,
        )
        sat = policy.saturation_score(snap)
        assert sat == pytest.approx(0.0)
