"""Tests for tuner/aimd.py — next_replicas and next_max_num_seqs, AIMD branches."""
from __future__ import annotations


from controller.tests.conftest import make_cfg
from controller.tuner.aimd import AimdTuner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tuner(**cfg_overrides: object) -> AimdTuner:
    return AimdTuner(make_cfg(**cfg_overrides))


# ---------------------------------------------------------------------------
# next_replicas — scale-out (high saturation)
# ---------------------------------------------------------------------------


def test_next_replicas_scaleout_adds_one() -> None:
    tuner = make_tuner(pressure_high=0.75, min_replicas=1, max_replicas=8)
    assert tuner.next_replicas(4, 1.0) == 5


def test_next_replicas_scaleout_at_exactly_pressure_high() -> None:
    tuner = make_tuner(pressure_high=0.75, min_replicas=1, max_replicas=8)
    assert tuner.next_replicas(4, 0.75) == 5


def test_next_replicas_scaleout_clamped_to_max_replicas() -> None:
    tuner = make_tuner(pressure_high=0.75, min_replicas=1, max_replicas=8)
    assert tuner.next_replicas(8, 1.0) == 8


def test_next_replicas_scaleout_below_max_increases() -> None:
    tuner = make_tuner(pressure_high=0.75, min_replicas=1, max_replicas=10)
    assert tuner.next_replicas(9, 1.0) == 10


# ---------------------------------------------------------------------------
# next_replicas — scale-in (low saturation)
# ---------------------------------------------------------------------------


def test_next_replicas_scalein_halves_current() -> None:
    tuner = make_tuner(pressure_low=0.35, min_replicas=1, max_replicas=8)
    assert tuner.next_replicas(8, 0.0) == 4


def test_next_replicas_scalein_at_exactly_pressure_low() -> None:
    tuner = make_tuner(pressure_low=0.35, min_replicas=1, max_replicas=8)
    assert tuner.next_replicas(8, 0.35) == 4


def test_next_replicas_scalein_clamped_to_min_replicas() -> None:
    tuner = make_tuner(pressure_low=0.35, min_replicas=2, max_replicas=8)
    assert tuner.next_replicas(2, 0.0) == 2  # 2//2=1 < min → clamped to 2


def test_next_replicas_scalein_floor_at_one() -> None:
    tuner = make_tuner(pressure_low=0.35, min_replicas=1, max_replicas=8)
    assert tuner.next_replicas(1, 0.0) == 1  # 1//2=0 < min → clamped to 1


# ---------------------------------------------------------------------------
# next_replicas — hold (mid saturation)
# ---------------------------------------------------------------------------


def test_next_replicas_hold_returns_current() -> None:
    tuner = make_tuner(pressure_low=0.35, pressure_high=0.75, min_replicas=1, max_replicas=8)
    assert tuner.next_replicas(5, 0.5) == 5


def test_next_replicas_hold_clamps_above_max() -> None:
    tuner = make_tuner(pressure_low=0.35, pressure_high=0.75, min_replicas=1, max_replicas=8)
    # current=20 in hold: clamped to max=8
    assert tuner.next_replicas(20, 0.5) == 8


def test_next_replicas_hold_clamps_below_min() -> None:
    tuner = make_tuner(pressure_low=0.35, pressure_high=0.75, min_replicas=3, max_replicas=8)
    # current=1 in hold: clamped to min=3
    assert tuner.next_replicas(1, 0.5) == 3


def test_next_replicas_hold_mid_saturation_above_low_below_high() -> None:
    tuner = make_tuner(pressure_low=0.35, pressure_high=0.75)
    # 0.35 < sat < 0.75 → hold
    assert tuner.next_replicas(4, 0.55) == 4


# ---------------------------------------------------------------------------
# next_replicas — result is always an int
# ---------------------------------------------------------------------------


def test_next_replicas_returns_int_type() -> None:
    tuner = make_tuner()
    result = tuner.next_replicas(4, 0.9)
    assert isinstance(result, int)


# ---------------------------------------------------------------------------
# next_max_num_seqs — scale-out
# ---------------------------------------------------------------------------


def test_next_max_num_seqs_scaleout_adds_128() -> None:
    tuner = make_tuner(pressure_high=0.75, min_max_num_seqs=128, max_max_num_seqs=2048)
    assert tuner.next_max_num_seqs(256, 1.0) == 384


def test_next_max_num_seqs_scaleout_at_exactly_pressure_high() -> None:
    tuner = make_tuner(pressure_high=0.75, min_max_num_seqs=128, max_max_num_seqs=2048)
    assert tuner.next_max_num_seqs(256, 0.75) == 384


def test_next_max_num_seqs_scaleout_clamped_to_max() -> None:
    tuner = make_tuner(pressure_high=0.75, min_max_num_seqs=128, max_max_num_seqs=2048)
    assert tuner.next_max_num_seqs(2048, 1.0) == 2048


def test_next_max_num_seqs_scaleout_near_max_clamped() -> None:
    tuner = make_tuner(pressure_high=0.75, min_max_num_seqs=128, max_max_num_seqs=2048)
    # 2000 + 128 = 2128 → clamped to 2048
    assert tuner.next_max_num_seqs(2000, 1.0) == 2048


# ---------------------------------------------------------------------------
# next_max_num_seqs — scale-in
# ---------------------------------------------------------------------------


def test_next_max_num_seqs_scalein_halves_current() -> None:
    tuner = make_tuner(pressure_low=0.35, min_max_num_seqs=128, max_max_num_seqs=2048)
    assert tuner.next_max_num_seqs(512, 0.0) == 256


def test_next_max_num_seqs_scalein_at_exactly_pressure_low() -> None:
    tuner = make_tuner(pressure_low=0.35, min_max_num_seqs=128, max_max_num_seqs=2048)
    assert tuner.next_max_num_seqs(512, 0.35) == 256


def test_next_max_num_seqs_scalein_clamped_to_min() -> None:
    tuner = make_tuner(pressure_low=0.35, min_max_num_seqs=128, max_max_num_seqs=2048)
    assert tuner.next_max_num_seqs(128, 0.0) == 128  # 128//2=64 < min → clamped


# ---------------------------------------------------------------------------
# next_max_num_seqs — hold
# ---------------------------------------------------------------------------


def test_next_max_num_seqs_hold_returns_current() -> None:
    tuner = make_tuner(pressure_low=0.35, pressure_high=0.75, min_max_num_seqs=128, max_max_num_seqs=2048)
    assert tuner.next_max_num_seqs(512, 0.5) == 512


def test_next_max_num_seqs_hold_clamps_above_max() -> None:
    tuner = make_tuner(pressure_low=0.35, pressure_high=0.75, min_max_num_seqs=128, max_max_num_seqs=2048)
    assert tuner.next_max_num_seqs(9999, 0.5) == 2048


def test_next_max_num_seqs_hold_clamps_below_min() -> None:
    tuner = make_tuner(pressure_low=0.35, pressure_high=0.75, min_max_num_seqs=128, max_max_num_seqs=2048)
    assert tuner.next_max_num_seqs(10, 0.5) == 128


# ---------------------------------------------------------------------------
# next_max_num_seqs — result is always an int
# ---------------------------------------------------------------------------


def test_next_max_num_seqs_returns_int_type() -> None:
    tuner = make_tuner()
    result = tuner.next_max_num_seqs(256, 0.9)
    assert isinstance(result, int)
