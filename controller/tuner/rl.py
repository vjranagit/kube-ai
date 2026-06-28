"""Tabular Q-learning tuner for kube-ai vLLM serving.

DIRECTION: HIGH saturation → scale OUT (add replicas / raise max_num_seqs).
This is the INVERSE of slurm-ai where high saturation → scale in.

Two independent Q-tables, one per tunable dimension:
  - 'replicas'     : bounds [cfg.min_replicas, cfg.max_replicas]
  - 'max_num_seqs' : bounds [cfg.min_max_num_seqs, cfg.max_max_num_seqs]

State space (per dimension): (pressure_bucket, current_value_bucket)
  - pressure_bucket : 5 equal-width bins over [0, 1]
  - value_bucket    : 6 bins linearly spaced across the configured [lo, hi] range

Action space:
  - ACTION_IN  (0) : scale in  — halve the current value, clamp to lo
  - ACTION_HOLD(1) : hold      — re-clamp to [lo, hi] in case bounds changed
  - ACTION_OUT (2) : scale out — additive step, clamp to hi

Fallback: AimdTuner for any state not present in the Q-table.
Persistence: JSON at cfg.rl_qtable_path (structure: {"replicas": {...}, "max_num_seqs": {...}}).
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from controller.tuner.aimd import AimdTuner

if TYPE_CHECKING:
    from controller.config import ControllerConfig

# ---------------------------------------------------------------------------
# Discretisation constants
# ---------------------------------------------------------------------------

_PRESSURE_BINS: int = 5
_VALUE_BINS: int = 6

ACTION_IN: int = 0
ACTION_HOLD: int = 1
ACTION_OUT: int = 2
N_ACTIONS: int = 3

# Additive scale-out steps (mirroring AimdTuner)
_REPLICAS_STEP: int = 1
_SEQS_STEP: int = 128

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

QTable = dict[tuple[int, int], list[float]]

# ---------------------------------------------------------------------------
# Discretisation helpers
# ---------------------------------------------------------------------------


def _pressure_bucket(saturation: float) -> int:
    """Map saturation ∈ [0, 1] → bucket ∈ [0, _PRESSURE_BINS-1]."""
    bucket = int(saturation * _PRESSURE_BINS)
    return min(bucket, _PRESSURE_BINS - 1)


def _value_bucket(current: int, lo: int, hi: int) -> int:
    """Map current ∈ [lo, hi] → bucket ∈ [0, _VALUE_BINS-1]."""
    if hi <= lo:
        return 0
    bucket = int((current - lo) / (hi - lo) * _VALUE_BINS)
    return min(bucket, _VALUE_BINS - 1)


def _apply_replicas_action(action: int, current: int, lo: int, hi: int) -> int:
    """Apply a replica-scaling action and clamp to [lo, hi].

    ACTION_IN:   max(lo, current // 2)  — mirrors AimdTuner scale-in
    ACTION_OUT:  min(hi, current + 1)   — mirrors AimdTuner scale-out
    ACTION_HOLD: clamp to [lo, hi]
    """
    if action == ACTION_IN:
        return max(lo, current // 2)
    if action == ACTION_OUT:
        return min(hi, current + _REPLICAS_STEP)
    return max(lo, min(hi, current))


def _apply_seqs_action(action: int, current: int, lo: int, hi: int) -> int:
    """Apply a max_num_seqs scaling action and clamp to [lo, hi].

    ACTION_IN:   max(lo, current // 2)   — mirrors AimdTuner scale-in
    ACTION_OUT:  min(hi, current + 128)  — mirrors AimdTuner scale-out
    ACTION_HOLD: clamp to [lo, hi]
    """
    if action == ACTION_IN:
        return max(lo, current // 2)
    if action == ACTION_OUT:
        return min(hi, current + _SEQS_STEP)
    return max(lo, min(hi, current))


# ---------------------------------------------------------------------------
# Q-table serialisation
# ---------------------------------------------------------------------------

# Storage format: {"replicas": {"pb": {"vb": [q0,q1,q2]}}, "max_num_seqs": {...}}
# JSON keys are always strings; convert on load/save.


def _qt_to_json(qtable: QTable) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    for (pb, vb), vals in qtable.items():
        out.setdefault(str(pb), {})[str(vb)] = list(vals)
    return out


def _json_to_qt(data: dict[str, dict[str, list[float]]]) -> QTable:
    qtable: QTable = {}
    for pb_str, inner in data.items():
        for vb_str, vals in inner.items():
            qtable[(int(pb_str), int(vb_str))] = list(vals)
    return qtable


def save_qtable(qtables: dict[str, QTable], path: str) -> None:
    """Persist both Q-tables to *path* as JSON, creating parent dirs as needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    data = {key: _qt_to_json(qt) for key, qt in qtables.items()}
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def load_qtable(path: str) -> dict[str, QTable]:
    """Load Q-tables from *path*.  Returns empty dict if file is missing/malformed."""
    try:
        with open(path) as fh:
            raw = json.load(fh)
        return {key: _json_to_qt(val) for key, val in raw.items()}
    except (FileNotFoundError, ValueError, KeyError):
        return {}


# ---------------------------------------------------------------------------
# RLTuner
# ---------------------------------------------------------------------------


class RLTuner:
    """Tabular Q-learning tuner with AIMD fallback for unseen states.

    Exposes the same interface as AimdTuner:
        next_replicas(current, saturation) -> int
        next_max_num_seqs(current, saturation) -> int

    Both methods stay within [min_replicas, max_replicas] and
    [min_max_num_seqs, max_max_num_seqs] respectively.

    Args:
        cfg: Controller configuration — provides bounds and RL hyper-parameters.
        qtables: Pre-loaded Q-tables dict; loaded from cfg.rl_qtable_path if None.
    """

    def __init__(
        self,
        cfg: "ControllerConfig",
        qtables: dict[str, QTable] | None = None,
    ) -> None:
        self.cfg = cfg
        self._fallback = AimdTuner(cfg)
        if qtables is not None:
            loaded = qtables
        else:
            loaded = load_qtable(cfg.rl_qtable_path)
        self._qt_replicas: QTable = loaded.get("replicas", {})
        self._qt_seqs: QTable = loaded.get("max_num_seqs", {})

    # ------------------------------------------------------------------
    # Public interface — identical to AimdTuner
    # ------------------------------------------------------------------

    def next_replicas(self, current: int, saturation: float) -> int:
        """Return next target replica count within [min_replicas, max_replicas].

        Uses greedy Q-policy for known states; falls back to AimdTuner otherwise.
        """
        lo, hi = self.cfg.min_replicas, self.cfg.max_replicas
        current = max(lo, min(hi, current))  # defensive clamp
        pb = _pressure_bucket(saturation)
        vb = _value_bucket(current, lo, hi)
        state = (pb, vb)
        if state in self._qt_replicas:
            action = int(max(range(N_ACTIONS), key=lambda a: self._qt_replicas[state][a]))
            result = _apply_replicas_action(action, current, lo, hi)
        else:
            result = self._fallback.next_replicas(current, saturation)
        return max(lo, min(hi, int(result)))

    def next_max_num_seqs(self, current: int, saturation: float) -> int:
        """Return next target max_num_seqs within [min_max_num_seqs, max_max_num_seqs].

        Uses greedy Q-policy for known states; falls back to AimdTuner otherwise.
        """
        lo, hi = self.cfg.min_max_num_seqs, self.cfg.max_max_num_seqs
        current = max(lo, min(hi, current))  # defensive clamp
        pb = _pressure_bucket(saturation)
        vb = _value_bucket(current, lo, hi)
        state = (pb, vb)
        if state in self._qt_seqs:
            action = int(max(range(N_ACTIONS), key=lambda a: self._qt_seqs[state][a]))
            result = _apply_seqs_action(action, current, lo, hi)
        else:
            result = self._fallback.next_max_num_seqs(current, saturation)
        return max(lo, min(hi, int(result)))

    # ------------------------------------------------------------------
    # Q-learning update (used during training; not called at inference)
    # ------------------------------------------------------------------

    def update(
        self,
        dim: str,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
    ) -> None:
        """Single Q-learning update (Bellman equation).

        Q(s,a) ← Q(s,a) + α * (r + γ * max_a' Q(s',a') − Q(s,a))

        Args:
            dim:        'replicas' or 'max_num_seqs'.
            state:      Current (pressure_bucket, value_bucket).
            action:     Action taken (ACTION_IN/HOLD/OUT).
            reward:     Observed scalar reward.
            next_state: Resulting (pressure_bucket, value_bucket).
        """
        alpha = self.cfg.rl_alpha
        gamma = self.cfg.rl_gamma
        qt = self._qt_replicas if dim == "replicas" else self._qt_seqs
        if state not in qt:
            qt[state] = [0.0] * N_ACTIONS
        if next_state not in qt:
            qt[next_state] = [0.0] * N_ACTIONS
        q_curr = qt[state][action]
        q_next_max = max(qt[next_state])
        qt[state][action] = q_curr + alpha * (reward + gamma * q_next_max - q_curr)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def qtables(self) -> dict[str, QTable]:
        """Live view of both Q-tables (not a copy — mutations are reflected)."""
        return {"replicas": self._qt_replicas, "max_num_seqs": self._qt_seqs}
