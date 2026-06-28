"""Training entry point for the kube-ai RL tuner.

Usage::

    python -m controller.tuner.train

Trains a Q-table using the built-in ServingSimulator and saves it to
cfg.rl_qtable_path (default: models/qtable.json).  Training is fully
deterministic via a fixed seed.

Hyperparameters are read from ControllerConfig (env vars / YAML):
    RL_TRAIN_EPISODES (default 300)
    RL_ALPHA          (default 0.1)
    RL_GAMMA          (default 0.9)
    RL_EPSILON        (default 0.1)
    RL_QTABLE_PATH    (default models/qtable.json)
"""

from __future__ import annotations

import os
import sys

from controller.config import ControllerConfig
from controller.tuner.rl import save_qtable
from controller.tuner.rl_env import train_rl


def main() -> None:
    cfg = ControllerConfig()

    print(
        f"Training RL tuner: episodes={cfg.rl_train_episodes} "
        f"alpha={cfg.rl_alpha} gamma={cfg.rl_gamma} epsilon={cfg.rl_epsilon}"
    )
    print(
        f"Bounds: replicas=[{cfg.min_replicas},{cfg.max_replicas}] "
        f"max_num_seqs=[{cfg.min_max_num_seqs},{cfg.max_max_num_seqs}]"
    )
    print(f"Output: {cfg.rl_qtable_path}")
    print()

    qtables = train_rl(cfg, episodes=cfg.rl_train_episodes, seed=42)

    n_r = len(qtables.get("replicas", {}))
    n_s = len(qtables.get("max_num_seqs", {}))
    print(f"Q-table states populated: replicas={n_r}  max_num_seqs={n_s}")

    # Ensure output directory exists
    out_dir = os.path.dirname(cfg.rl_qtable_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_qtable(qtables, cfg.rl_qtable_path)
    print(f"Q-table saved to {cfg.rl_qtable_path}")

    if n_r == 0 and n_s == 0:
        print(
            "WARNING: both Q-tables are empty — check simulator / hyperparameters.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
