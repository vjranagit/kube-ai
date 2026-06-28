"""AimdTuner — Additive-Increase / Multiplicative-Decrease for Kubernetes vLLM serving.

⚠️ DIRECTION INVERSION vs slurm-ai ⚠️
High saturation → SCALE OUT (increase replicas / raise max_num_seqs).
Low saturation  → scale in  (decrease replicas / lower max_num_seqs).

Replica AIMD:
    scale-out: current + 1       (additive, conservative — adding a pod is expensive)
    scale-in:  current // 2      (multiplicative, aggressive — shed idle capacity quickly)

max_num_seqs AIMD:
    scale-out: current + 128     (larger step; no pod restart cost)
    scale-in:  max(min_bound, current // 2)

Both methods return values clamped to their respective [min, max] bounds.
Bounds are enforced here AND in K8sActuator.apply() (double-clamped by design).
"""

from controller.config import ControllerConfig


class AimdTuner:
    def __init__(self, cfg: ControllerConfig) -> None:
        self.cfg = cfg

    def next_replicas(self, current: int, saturation: float) -> int:
        """Compute next target replica count.

        High saturation → +1 (scale out).
        Low saturation  → //2 (scale in).
        Mid range       → hold (clamp to bounds).
        """
        if saturation >= self.cfg.pressure_high:
            # Scale out: additive increase
            next_val = min(self.cfg.max_replicas, current + 1)
        elif saturation <= self.cfg.pressure_low:
            # Scale in: multiplicative decrease
            next_val = max(self.cfg.min_replicas, current // 2)
        else:
            # Hold: re-clamp current in case bounds changed
            next_val = max(self.cfg.min_replicas, min(self.cfg.max_replicas, current))
        return next_val

    def next_max_num_seqs(self, current: int, saturation: float) -> int:
        """Compute next target max_num_seqs value.

        High saturation → +128 (scale out).
        Low saturation  → //2 (scale in).
        Mid range       → hold.
        """
        if saturation >= self.cfg.pressure_high:
            next_val = min(self.cfg.max_max_num_seqs, current + 128)
        elif saturation <= self.cfg.pressure_low:
            next_val = max(self.cfg.min_max_num_seqs, current // 2)
        else:
            next_val = max(self.cfg.min_max_num_seqs, min(self.cfg.max_max_num_seqs, current))
        return next_val
