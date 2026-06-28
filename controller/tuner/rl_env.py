"""Deterministic vLLM serving simulator and Q-learning trainer for RLTuner.

The simulator models a Kubernetes-hosted vLLM inference cluster with:
  - Poisson arrivals (seeded, pure stdlib — no numpy)
  - Service rate proportional to replicas × max_num_seqs
  - Saturation scored using the SAME formula as PolicyEngine (policy/engine.py)

Reward signal per step:
    reward = throughput − saturation_penalty − replica_cost − thrash_penalty

    throughput        = completions / arrival_rate   (normalized, rewards serving load)
    saturation_penalty= saturation * 0.5             (penalises high pressure)
    replica_cost      = norm_replicas * 0.10         (penalises unnecessary capacity)
    thrash_penalty    = (|Δreplicas/range_r| + |Δseqs/range_s|) * 0.05

This combination rewards scaling out under high saturation (matches kube-ai's direction)
and scaling in when load is light (cost recovery).
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from controller.tuner.rl import (
    N_ACTIONS,
    QTable,
    _apply_replicas_action,
    _apply_seqs_action,
    _pressure_bucket,
    _value_bucket,
    save_qtable,
)

if TYPE_CHECKING:
    from controller.config import ControllerConfig

# ---------------------------------------------------------------------------
# Poisson sampler (stdlib only)
# ---------------------------------------------------------------------------


def _poisson(rng: random.Random, lam: float) -> int:
    """Draw a Poisson sample via Knuth's algorithm (pure stdlib)."""
    if lam <= 0:
        return 0
    l_ = math.exp(-lam)
    k = 0
    p = 1.0
    while p > l_:
        p *= rng.random()
        k += 1
    return k - 1


# ---------------------------------------------------------------------------
# Serving simulator
# ---------------------------------------------------------------------------


class ServingSimulator:
    """Seeded vLLM-style serving simulator.

    Models Poisson arrivals with a serving cluster whose throughput scales with
    replicas × max_num_seqs.  Saturation is computed using PolicyEngine's formula:

        saturation = 0.50 * queue_pressure
                   + 0.30 * cache_pressure
                   + 0.20 * latency_pressure

    Args:
        arrival_rate: Mean requests arriving per simulation tick (Poisson λ).
        slo_sec:      TTFT SLO in seconds (used for latency_pressure).
        seed:         RNG seed for full reproducibility.
    """

    _P_DONE: float = 0.4  # probability a running request completes each tick

    def __init__(
        self,
        arrival_rate: float = 8.0,
        slo_sec: float = 2.0,
        seed: int = 42,
    ) -> None:
        self._arrival_rate = arrival_rate
        self._slo_sec = slo_sec
        self._rng = random.Random(seed)
        self._waiting: int = 0
        self._running: int = 0

    def reset(self) -> None:
        """Reset queue state (RNG is NOT reset — create a new instance for that)."""
        self._waiting = 0
        self._running = 0

    def step(self, replicas: int, max_num_seqs: int) -> tuple[float, float]:
        """Advance simulation by one tick.

        Service capacity is replicas × (max_num_seqs // 128) so the range
        [1×128 .. 8×2048] maps to [1 .. 128] effective slots — tractable for
        a Poisson model with arrival_rate ≤ 15.

        Args:
            replicas:     Current replica count.
            max_num_seqs: Current max-sequences-per-forward-pass setting.

        Returns:
            (saturation, throughput) both in [0, 1].
            saturation uses the PolicyEngine formula.
            throughput  = completions / arrival_rate (capped at 1).
        """
        capacity = max(1, replicas * (max_num_seqs // 128))

        # Arrivals
        arrivals = _poisson(self._rng, self._arrival_rate)
        self._waiting += arrivals

        # Completions (each running unit finishes with p = _P_DONE)
        completions = sum(
            1 for _ in range(self._running) if self._rng.random() < self._P_DONE
        )
        self._running = max(0, self._running - completions)

        # Admit new requests up to capacity
        admissible = max(0, capacity - self._running)
        admitted = min(self._waiting, admissible)
        self._waiting -= admitted
        self._running += admitted

        # Sub-scores (PolicyEngine formula)
        total = max(1, self._waiting + self._running)
        queue_pressure = self._waiting / total

        kv_cache = min(1.0, self._running / capacity)
        cache_pressure = kv_cache

        p95_ttft = self._slo_sec * (1.0 + self._waiting / max(1.0, float(capacity)))
        latency_pressure = max(0.0, min(1.0, (p95_ttft - self._slo_sec) / self._slo_sec))

        saturation = (
            0.50 * queue_pressure
            + 0.30 * cache_pressure
            + 0.20 * latency_pressure
        )
        saturation = min(1.0, saturation)

        throughput = min(1.0, completions / max(1.0, self._arrival_rate))
        return saturation, throughput


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def _reward(
    saturation: float,
    throughput: float,
    replicas: int,
    prev_replicas: int,
    max_num_seqs: int,
    prev_seqs: int,
    cfg: "ControllerConfig",
) -> float:
    """Compute per-step reward.

    Rewards throughput; penalises saturation, unnecessary replica use, and thrashing.
    """
    range_r = max(1, cfg.max_replicas - cfg.min_replicas)
    range_s = max(1, cfg.max_max_num_seqs - cfg.min_max_num_seqs)

    sat_penalty = saturation * 0.5
    replica_cost = (replicas - cfg.min_replicas) / range_r * 0.10
    thrash = (
        abs(replicas - prev_replicas) / range_r
        + abs(max_num_seqs - prev_seqs) / range_s
    ) * 0.05
    return throughput - sat_penalty - replica_cost - thrash


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_rl(
    cfg: "ControllerConfig",
    episodes: int | None = None,
    seed: int = 42,
) -> dict[str, QTable]:
    """Train Q-tables for replicas and max_num_seqs via tabular Q-learning.

    Each episode uses a freshly seeded ServingSimulator with a randomly drawn
    arrival rate, exposing the agent to diverse load conditions.

    Args:
        cfg:      Controller config — provides bounds and RL hyper-parameters.
        episodes: Training episodes; defaults to cfg.rl_train_episodes.
        seed:     Master seed for full reproducibility.

    Returns:
        dict with keys 'replicas' and 'max_num_seqs', each a trained QTable.
    """
    from controller.tuner.rl import RLTuner

    if episodes is None:
        episodes = cfg.rl_train_episodes

    min_r, max_r = cfg.min_replicas, cfg.max_replicas
    min_s, max_s = cfg.min_max_num_seqs, cfg.max_max_num_seqs
    epsilon = cfg.rl_epsilon
    steps_per_episode = 200

    # Arrival rate range: from light load to 2× minimum capacity
    min_cap = min_r * (min_s // 128)  # e.g. 1 * 1 = 1 at kube-ai defaults
    arr_lo = max(1.0, min_cap * 0.5)
    arr_hi = max(arr_lo + 1.0, min_cap * 8.0 + 4.0)  # ensures variety

    master_rng = random.Random(seed)
    tuner = RLTuner(cfg, qtables={"replicas": {}, "max_num_seqs": {}})

    for _ep in range(episodes):
        ep_seed = master_rng.randint(0, 2**31)
        arrival_rate = master_rng.uniform(arr_lo, arr_hi)
        sim = ServingSimulator(
            arrival_rate=arrival_rate,
            slo_sec=cfg.ttft_slo_sec,
            seed=ep_seed,
        )

        replicas = (min_r + max_r) // 2
        max_num_seqs = (min_s + max_s) // 2
        prev_replicas = replicas
        prev_seqs = max_num_seqs

        sat, _throughput = sim.step(replicas, max_num_seqs)

        for _ in range(steps_per_episode):
            pb = _pressure_bucket(sat)
            vb_r = _value_bucket(replicas, min_r, max_r)
            vb_s = _value_bucket(max_num_seqs, min_s, max_s)
            state_r = (pb, vb_r)
            state_s = (pb, vb_s)

            qt_r = tuner.qtables["replicas"]
            qt_s = tuner.qtables["max_num_seqs"]

            # Epsilon-greedy action selection (replicas)
            if master_rng.random() < epsilon or state_r not in qt_r:
                action_r = master_rng.randint(0, N_ACTIONS - 1)
            else:
                action_r = int(max(range(N_ACTIONS), key=lambda a: qt_r[state_r][a]))

            # Epsilon-greedy action selection (max_num_seqs)
            if master_rng.random() < epsilon or state_s not in qt_s:
                action_s = master_rng.randint(0, N_ACTIONS - 1)
            else:
                action_s = int(max(range(N_ACTIONS), key=lambda a: qt_s[state_s][a]))

            new_replicas = _apply_replicas_action(action_r, replicas, min_r, max_r)
            new_seqs = _apply_seqs_action(action_s, max_num_seqs, min_s, max_s)

            next_sat, next_throughput = sim.step(new_replicas, new_seqs)

            r = _reward(next_sat, next_throughput, new_replicas, prev_replicas, new_seqs, prev_seqs, cfg)

            next_pb = _pressure_bucket(next_sat)
            next_vb_r = _value_bucket(new_replicas, min_r, max_r)
            next_vb_s = _value_bucket(new_seqs, min_s, max_s)
            next_state_r = (next_pb, next_vb_r)
            next_state_s = (next_pb, next_vb_s)

            tuner.update("replicas", state_r, action_r, r, next_state_r)
            tuner.update("max_num_seqs", state_s, action_s, r, next_state_s)

            prev_replicas = replicas
            prev_seqs = max_num_seqs
            replicas = new_replicas
            max_num_seqs = new_seqs
            sat = next_sat

    return tuner.qtables


# ---------------------------------------------------------------------------
# Persistence helpers (thin wrappers; canonical implementations live in rl.py)
# ---------------------------------------------------------------------------


def save_qtable_json(qtables: dict[str, QTable], path: str) -> None:
    """Save Q-tables to *path* (creates parent dirs)."""
    save_qtable(qtables, path)
