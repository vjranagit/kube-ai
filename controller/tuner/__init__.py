"""Tuner package — exports AimdTuner and the build_tuner factory.

build_tuner(cfg) returns an AimdTuner for tuner_kind='aimd'.
For tuner_kind='rl', it also returns an AimdTuner with a TODO — RL lands in commit 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from controller.tuner.aimd import AimdTuner

if TYPE_CHECKING:
    from controller.config import ControllerConfig

Tuner = Union[AimdTuner]  # Union will widen when RLTuner is added in commit 3

__all__ = ["AimdTuner", "Tuner", "build_tuner"]


def build_tuner(cfg: "ControllerConfig") -> AimdTuner:
    """Factory: return the appropriate tuner for cfg.tuner_kind.

    commit 1: always returns AimdTuner.
    TODO (commit 3): import RLTuner and return it when tuner_kind == 'rl'.
    """
    if cfg.tuner_kind.lower() == "rl":
        # TODO (commit 3): return RLTuner(cfg) once rl.py is implemented
        import logging

        logging.getLogger("kube-ai.tuner").warning(
            "tuner_kind=rl requested but RL tuner is not yet implemented (commit 3); "
            "falling back to AimdTuner"
        )
    return AimdTuner(cfg)
