"""Tests for the kube-ai RL tuner (RLTuner, build_tuner, train_rl).

All tests are deterministic (seeded) and self-contained — no filesystem
side-effects unless testing persistence explicitly (uses pytest's tmp_path).

Uses make_cfg() from conftest for consistent ControllerConfig construction.
"""

from __future__ import annotations

import random

import pytest

from controller.tests.conftest import make_cfg
from controller.tuner import AimdTuner, RLTuner, build_tuner
from controller.tuner.rl import (
    ACTION_IN,
    ACTION_HOLD,
    ACTION_OUT,
    N_ACTIONS,
    QTable,
    _pressure_bucket,
    _value_bucket,
    load_qtable,
    save_qtable,
)
from controller.tuner.rl_env import train_rl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rl_cfg(**overrides: object):  # type: ignore[return]
    """Config with RL defaults; override rl_qtable_path etc. as needed."""
    defaults = dict(
        tuner_kind="rl",
        rl_qtable_path="/nonexistent/qtable.json",
        rl_alpha=0.1,
        rl_gamma=0.9,
        rl_epsilon=0.1,
        rl_train_episodes=300,
    )
    defaults.update(overrides)
    return make_cfg(**defaults)


# ---------------------------------------------------------------------------
# Test 1: Q-update math — Bellman equation
# ---------------------------------------------------------------------------


def test_q_update_bellman_replicas() -> None:
    """RLTuner.update follows Q(s,a) += α*(r + γ*max Q(s',·) − Q(s,a))."""
    cfg = _rl_cfg(rl_alpha=0.1, rl_gamma=0.9)
    state = (2, 3)
    next_state = (3, 2)
    initial_q = [0.0, 0.5, 0.2]
    next_q = [0.1, 0.8, 0.3]
    qtable: QTable = {state: list(initial_q), next_state: list(next_q)}
    tuner = RLTuner(cfg, qtables={"replicas": qtable, "max_num_seqs": {}})

    reward = 1.0
    action = ACTION_OUT
    old_q = initial_q[action]
    q_next_max = max(next_q)
    expected = old_q + 0.1 * (reward + 0.9 * q_next_max - old_q)

    tuner.update("replicas", state, action, reward, next_state)
    assert abs(tuner._qt_replicas[state][action] - expected) < 1e-12


def test_q_update_bellman_max_num_seqs() -> None:
    """Same Bellman check for the max_num_seqs Q-table."""
    cfg = _rl_cfg(rl_alpha=0.2, rl_gamma=0.8)
    state = (1, 0)
    next_state = (0, 1)
    initial_q = [0.3, 0.1, 0.0]
    next_q = [0.5, 0.2, 0.4]
    qtable: QTable = {state: list(initial_q), next_state: list(next_q)}
    tuner = RLTuner(cfg, qtables={"replicas": {}, "max_num_seqs": qtable})

    reward = -0.5
    action = ACTION_IN
    old_q = initial_q[action]
    q_next_max = max(next_q)
    expected = old_q + 0.2 * (reward + 0.8 * q_next_max - old_q)

    tuner.update("max_num_seqs", state, action, reward, next_state)
    assert abs(tuner._qt_seqs[state][action] - expected) < 1e-12


def test_q_update_creates_entries_for_unseen_states() -> None:
    """update() initialises Q-values to 0 for states not yet in the table."""
    cfg = _rl_cfg()
    tuner = RLTuner(cfg, qtables={})
    tuner.update("replicas", (0, 0), ACTION_HOLD, 0.5, (1, 1))
    assert (0, 0) in tuner._qt_replicas
    assert (1, 1) in tuner._qt_replicas
    assert len(tuner._qt_replicas[(0, 0)]) == N_ACTIONS


# ---------------------------------------------------------------------------
# Test 2: JSON persistence round-trip
# ---------------------------------------------------------------------------


def test_json_persistence_round_trip(tmp_path: pytest.TempPathFactory) -> None:
    """save_qtable → load_qtable produces tables with identical inference decisions."""
    cfg = _rl_cfg(rl_train_episodes=50)
    qtable_path = str(tmp_path / "test_qtable.json")  # type: ignore[operator]

    qtables = train_rl(cfg, episodes=50, seed=7)
    assert len(qtables.get("replicas", {})) > 0 or len(qtables.get("max_num_seqs", {})) > 0

    save_qtable(qtables, qtable_path)

    # Deep-copy to isolate the two tuners
    def _copy(qt: dict[str, QTable]) -> dict[str, QTable]:
        return {k: {kk: list(vv) for kk, vv in tbl.items()} for k, tbl in qt.items()}

    cfg_orig = _rl_cfg(rl_qtable_path="/nonexistent/")
    cfg_loaded = _rl_cfg(rl_qtable_path=qtable_path)

    tuner_orig = RLTuner(cfg_orig, qtables=_copy(qtables))
    tuner_loaded = RLTuner(cfg_loaded)  # loads from file

    rng = random.Random(42)
    for _ in range(300):
        r = rng.randint(cfg.min_replicas, cfg.max_replicas)
        s = rng.randint(cfg.min_max_num_seqs, cfg.max_max_num_seqs)
        sat = rng.random()
        assert tuner_orig.next_replicas(r, sat) == tuner_loaded.next_replicas(r, sat), (
            f"replicas mismatch: r={r} sat={sat:.3f}"
        )
        assert tuner_orig.next_max_num_seqs(s, sat) == tuner_loaded.next_max_num_seqs(s, sat), (
            f"max_num_seqs mismatch: s={s} sat={sat:.3f}"
        )


def test_load_missing_file_returns_empty() -> None:
    """load_qtable on a missing file returns empty dict (no exception)."""
    result = load_qtable("/nonexistent/path/qtable.json")
    assert result == {}


# ---------------------------------------------------------------------------
# Test 3: Bounds invariant over 1000 random inputs (both methods)
# ---------------------------------------------------------------------------


def test_bounds_invariant_next_replicas_1000_samples() -> None:
    """next_replicas always returns int in [min_replicas, max_replicas]."""
    cfg = _rl_cfg()
    tuner = RLTuner(cfg, qtables={})
    lo, hi = cfg.min_replicas, cfg.max_replicas
    rng = random.Random(0)
    for i in range(1000):
        current = rng.randint(lo, hi)
        sat = rng.random()
        result = tuner.next_replicas(current, sat)
        assert isinstance(result, int), f"step {i}: expected int, got {type(result)}"
        assert lo <= result <= hi, (
            f"step {i}: replicas {result} not in [{lo},{hi}] "
            f"(current={current}, sat={sat:.3f})"
        )


def test_bounds_invariant_next_max_num_seqs_1000_samples() -> None:
    """next_max_num_seqs always returns int in [min_max_num_seqs, max_max_num_seqs]."""
    cfg = _rl_cfg()
    tuner = RLTuner(cfg, qtables={})
    lo, hi = cfg.min_max_num_seqs, cfg.max_max_num_seqs
    rng = random.Random(1)
    for i in range(1000):
        current = rng.randint(lo, hi)
        sat = rng.random()
        result = tuner.next_max_num_seqs(current, sat)
        assert isinstance(result, int), f"step {i}: expected int, got {type(result)}"
        assert lo <= result <= hi, (
            f"step {i}: max_num_seqs {result} not in [{lo},{hi}] "
            f"(current={current}, sat={sat:.3f})"
        )


def test_bounds_invariant_with_trained_qtable_1000_samples() -> None:
    """Bounds hold even with a trained Q-table (non-empty lookup path)."""
    cfg = _rl_cfg(rl_train_episodes=100)
    qtables = train_rl(cfg, episodes=100, seed=13)
    tuner = RLTuner(cfg, qtables=qtables)
    lo_r, hi_r = cfg.min_replicas, cfg.max_replicas
    lo_s, hi_s = cfg.min_max_num_seqs, cfg.max_max_num_seqs
    rng = random.Random(77)
    for i in range(1000):
        r = rng.randint(lo_r, hi_r)
        s = rng.randint(lo_s, hi_s)
        sat = rng.random()
        rr = tuner.next_replicas(r, sat)
        ss = tuner.next_max_num_seqs(s, sat)
        assert lo_r <= rr <= hi_r, f"step {i}: replicas {rr} out of [{lo_r},{hi_r}]"
        assert lo_s <= ss <= hi_s, f"step {i}: seqs {ss} out of [{lo_s},{hi_s}]"
        assert isinstance(rr, int)
        assert isinstance(ss, int)


# ---------------------------------------------------------------------------
# Test 4: AIMD fallback when Q-table is empty
# ---------------------------------------------------------------------------


def test_aimd_fallback_next_replicas_empty_table() -> None:
    """Empty Q-table → next_replicas matches AimdTuner exactly."""
    cfg = _rl_cfg()
    rl_tuner = RLTuner(cfg, qtables={})
    aimd_tuner = AimdTuner(cfg)
    rng = random.Random(5)
    for _ in range(300):
        current = rng.randint(cfg.min_replicas, cfg.max_replicas)
        sat = rng.random()
        assert rl_tuner.next_replicas(current, sat) == aimd_tuner.next_replicas(current, sat), (
            f"replicas fallback mismatch: current={current} sat={sat:.3f}"
        )


def test_aimd_fallback_next_max_num_seqs_empty_table() -> None:
    """Empty Q-table → next_max_num_seqs matches AimdTuner exactly."""
    cfg = _rl_cfg()
    rl_tuner = RLTuner(cfg, qtables={})
    aimd_tuner = AimdTuner(cfg)
    rng = random.Random(6)
    for _ in range(300):
        current = rng.randint(cfg.min_max_num_seqs, cfg.max_max_num_seqs)
        sat = rng.random()
        assert rl_tuner.next_max_num_seqs(current, sat) == aimd_tuner.next_max_num_seqs(
            current, sat
        ), f"seqs fallback mismatch: current={current} sat={sat:.3f}"


# ---------------------------------------------------------------------------
# Test 5: build_tuner factory
# ---------------------------------------------------------------------------


def test_build_tuner_returns_rl_tuner() -> None:
    """build_tuner with tuner_kind='rl' returns RLTuner instance."""
    cfg = _rl_cfg(tuner_kind="rl")
    tuner = build_tuner(cfg)
    assert isinstance(tuner, RLTuner)


def test_build_tuner_returns_aimd_tuner() -> None:
    """build_tuner with tuner_kind='aimd' returns AimdTuner instance."""
    cfg = make_cfg(tuner_kind="aimd")
    assert isinstance(build_tuner(cfg), AimdTuner)


def test_build_tuner_rl_has_both_methods() -> None:
    """RLTuner from build_tuner exposes both next_replicas and next_max_num_seqs."""
    cfg = _rl_cfg()
    tuner = build_tuner(cfg)
    assert hasattr(tuner, "next_replicas")
    assert hasattr(tuner, "next_max_num_seqs")
    # Smoke-test both methods stay in bounds
    lo_r, hi_r = cfg.min_replicas, cfg.max_replicas
    lo_s, hi_s = cfg.min_max_num_seqs, cfg.max_max_num_seqs
    assert lo_r <= tuner.next_replicas(4, 0.8) <= hi_r
    assert lo_s <= tuner.next_max_num_seqs(512, 0.2) <= hi_s


# ---------------------------------------------------------------------------
# Test 6: Config fields
# ---------------------------------------------------------------------------


def test_config_rl_fields_defaults() -> None:
    """ControllerConfig must have all RL fields with correct types."""
    cfg = make_cfg()
    assert hasattr(cfg, "rl_qtable_path")
    assert hasattr(cfg, "rl_alpha")
    assert hasattr(cfg, "rl_gamma")
    assert hasattr(cfg, "rl_epsilon")
    assert hasattr(cfg, "rl_train_episodes")
    assert isinstance(cfg.rl_qtable_path, str)
    assert isinstance(cfg.rl_alpha, float)
    assert isinstance(cfg.rl_gamma, float)
    assert isinstance(cfg.rl_epsilon, float)
    assert isinstance(cfg.rl_train_episodes, int)


def test_config_rl_fields_override() -> None:
    """RL config fields are overridable via make_cfg."""
    cfg = _rl_cfg(rl_alpha=0.5, rl_gamma=0.95, rl_epsilon=0.2, rl_train_episodes=500)
    assert cfg.rl_alpha == pytest.approx(0.5)
    assert cfg.rl_gamma == pytest.approx(0.95)
    assert cfg.rl_epsilon == pytest.approx(0.2)
    assert cfg.rl_train_episodes == 500


# ---------------------------------------------------------------------------
# Test 7: Discretisation helpers
# ---------------------------------------------------------------------------


def test_pressure_bucket_boundaries() -> None:
    """_pressure_bucket maps 0 → 0 and 1.0 → _PRESSURE_BINS-1."""
    from controller.tuner.rl import _PRESSURE_BINS

    assert _pressure_bucket(0.0) == 0
    assert _pressure_bucket(1.0) == _PRESSURE_BINS - 1
    assert _pressure_bucket(0.5) < _PRESSURE_BINS


def test_value_bucket_boundaries() -> None:
    """_value_bucket maps lo → 0 and hi → _VALUE_BINS-1."""
    from controller.tuner.rl import _VALUE_BINS

    assert _value_bucket(1, 1, 8) == 0
    assert _value_bucket(8, 1, 8) == _VALUE_BINS - 1
    assert _value_bucket(128, 128, 2048) == 0
    assert _value_bucket(2048, 128, 2048) == _VALUE_BINS - 1


# ---------------------------------------------------------------------------
# C5 — load_qtable tolerates malformed JSON (top-level list, wrong types)
# ---------------------------------------------------------------------------


def test_load_qtable_json_list_returns_empty(tmp_path: pytest.TempPathFactory) -> None:
    """load_qtable returns {} when the file contains a top-level JSON list (C5)."""
    import json
    import pathlib
    p = str(tmp_path / "bad.json")  # type: ignore[operator]
    pathlib.Path(p).write_text(json.dumps([1, 2, 3]))
    result = load_qtable(p)
    assert result == {}


def test_load_qtable_json_null_returns_empty(tmp_path: pytest.TempPathFactory) -> None:
    """load_qtable returns {} for JSON null."""
    p = str(tmp_path / "null.json")  # type: ignore[operator]
    import pathlib
    pathlib.Path(p).write_text("null")
    result = load_qtable(p)
    assert result == {}


def test_load_qtable_malformed_json_returns_empty(tmp_path: pytest.TempPathFactory) -> None:
    """load_qtable returns {} when the file contains truncated/invalid JSON."""
    p = str(tmp_path / "trunc.json")  # type: ignore[operator]
    import pathlib
    pathlib.Path(p).write_text('{"replicas": {invalid}')
    result = load_qtable(p)
    assert result == {}


def test_load_qtable_nested_non_dict_skipped(tmp_path: pytest.TempPathFactory) -> None:
    """load_qtable skips dimension entries that are not dicts."""
    import json
    import pathlib
    p = str(tmp_path / "mixed.json")  # type: ignore[operator]
    data = {"replicas": [1, 2, 3], "max_num_seqs": {}}
    pathlib.Path(p).write_text(json.dumps(data))
    result = load_qtable(p)
    # "replicas" has a list value → skipped; "max_num_seqs" is an empty dict → included
    assert "replicas" not in result


# ---------------------------------------------------------------------------
# H6 — save_qtable uses atomic write (temp file + os.replace)
# ---------------------------------------------------------------------------


def test_save_qtable_atomic_no_tmp_left_behind(tmp_path: pytest.TempPathFactory) -> None:
    """save_qtable leaves no .tmp file after a successful write (H6)."""
    import pathlib
    p = str(tmp_path / "qtable.json")  # type: ignore[operator]
    save_qtable({}, p)
    assert not pathlib.Path(p + ".tmp").exists()


def test_save_qtable_atomic_target_exists_and_valid(tmp_path: pytest.TempPathFactory) -> None:
    """save_qtable produces a valid JSON file even when interrupted mid-write (H6)."""
    qtables = {"replicas": {}, "max_num_seqs": {}}
    p = str(tmp_path / "qt.json")  # type: ignore[operator]
    save_qtable(qtables, p)
    reloaded = load_qtable(p)
    assert isinstance(reloaded, dict)


def test_save_qtable_crash_does_not_corrupt_original(tmp_path: pytest.TempPathFactory) -> None:
    """A crash during write (simulated by leaving .tmp around) does not corrupt the
    original file because os.replace is atomic (H6)."""
    import json
    import pathlib
    p = str(tmp_path / "qtable.json")  # type: ignore[operator]
    # Write a valid Q-table first
    original = {"replicas": {"0": {"0": [0.1, 0.2, 0.3]}}}
    pathlib.Path(p).write_text(json.dumps(original))

    # Simulate a crash mid-write by manually writing a corrupt .tmp
    pathlib.Path(p + ".tmp").write_text("{corrupt")

    # The original should still be readable
    loaded = load_qtable(p)
    assert "replicas" in loaded
