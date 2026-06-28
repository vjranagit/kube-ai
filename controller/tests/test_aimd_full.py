"""1 000-iteration invariant tests for AimdTuner — both output methods."""
from __future__ import annotations

import random

from controller.tests.conftest import make_cfg
from controller.tuner.aimd import AimdTuner

ITERATIONS = 1000
SEED = 42


# ---------------------------------------------------------------------------
# next_replicas invariants
# ---------------------------------------------------------------------------


def test_next_replicas_always_within_bounds() -> None:
    cfg = make_cfg(min_replicas=1, max_replicas=8)
    tuner = AimdTuner(cfg)
    rng = random.Random(SEED)
    current = 4
    for _ in range(ITERATIONS):
        sat = rng.random()
        nxt = tuner.next_replicas(current, sat)
        assert cfg.min_replicas <= nxt <= cfg.max_replicas, (
            f"next_replicas={nxt} outside [{cfg.min_replicas}, {cfg.max_replicas}]"
        )
        current = nxt


def test_next_replicas_always_int() -> None:
    cfg = make_cfg()
    tuner = AimdTuner(cfg)
    rng = random.Random(SEED)
    current = 4
    for _ in range(ITERATIONS):
        sat = rng.random()
        nxt = tuner.next_replicas(current, sat)
        assert isinstance(nxt, int)
        current = nxt


def test_next_replicas_never_raises() -> None:
    cfg = make_cfg()
    tuner = AimdTuner(cfg)
    rng = random.Random(SEED)
    current = 4
    for _ in range(ITERATIONS):
        sat = rng.random()
        current = tuner.next_replicas(current, sat)


def test_next_replicas_custom_bounds_respected() -> None:
    cfg = make_cfg(min_replicas=3, max_replicas=20)
    tuner = AimdTuner(cfg)
    rng = random.Random(SEED)
    current = 10
    for _ in range(ITERATIONS):
        sat = rng.random()
        nxt = tuner.next_replicas(current, sat)
        assert 3 <= nxt <= 20
        current = nxt


# ---------------------------------------------------------------------------
# next_max_num_seqs invariants
# ---------------------------------------------------------------------------


def test_next_max_num_seqs_always_within_bounds() -> None:
    cfg = make_cfg(min_max_num_seqs=128, max_max_num_seqs=2048)
    tuner = AimdTuner(cfg)
    rng = random.Random(SEED)
    current = 512
    for _ in range(ITERATIONS):
        sat = rng.random()
        nxt = tuner.next_max_num_seqs(current, sat)
        assert cfg.min_max_num_seqs <= nxt <= cfg.max_max_num_seqs, (
            f"next_max_num_seqs={nxt} outside "
            f"[{cfg.min_max_num_seqs}, {cfg.max_max_num_seqs}]"
        )
        current = nxt


def test_next_max_num_seqs_always_int() -> None:
    cfg = make_cfg()
    tuner = AimdTuner(cfg)
    rng = random.Random(SEED)
    current = 512
    for _ in range(ITERATIONS):
        sat = rng.random()
        nxt = tuner.next_max_num_seqs(current, sat)
        assert isinstance(nxt, int)
        current = nxt


# ---------------------------------------------------------------------------
# Convergence: high saturation trends DOWN toward min (scale-in direction inverted)
# ---------------------------------------------------------------------------


def test_sustained_low_saturation_replicas_trend_to_min() -> None:
    """Sustained low saturation → scale-in → replicas converge toward min_replicas."""
    cfg = make_cfg(min_replicas=1, max_replicas=8, pressure_low=0.35)
    tuner = AimdTuner(cfg)
    current = 8
    for _ in range(50):  # 8→4→2→1 in 3 steps; 50 is plenty
        current = tuner.next_replicas(current, 0.0)
    assert current == cfg.min_replicas


def test_sustained_high_saturation_replicas_trend_to_max() -> None:
    """Sustained high saturation → scale-out → replicas converge toward max_replicas."""
    cfg = make_cfg(min_replicas=1, max_replicas=8, pressure_high=0.75)
    tuner = AimdTuner(cfg)
    current = 1
    for _ in range(20):  # 1→2→3→...→8 in 7 steps; 20 is plenty
        current = tuner.next_replicas(current, 1.0)
    assert current == cfg.max_replicas


def test_sustained_low_saturation_max_num_seqs_trend_to_min() -> None:
    cfg = make_cfg(min_max_num_seqs=128, max_max_num_seqs=2048, pressure_low=0.35)
    tuner = AimdTuner(cfg)
    current = 2048
    for _ in range(20):  # 2048 → 1024 → 512 → 256 → 128 in 4 steps; 20 plenty
        current = tuner.next_max_num_seqs(current, 0.0)
    assert current == cfg.min_max_num_seqs
