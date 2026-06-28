"""K8sActuator — applies PolicyDecision to the Kubernetes cluster.

Safety invariants enforced here (do not remove):
  - dry_run=True by default; only explicit override allows live mutation.
  - Bounds double-clamped: tuner already clamps, actuator re-clamps regardless.
  - Per-path cooldowns:
      replicas  → cooldown_sec     (default 60 s)
      params    → param_cooldown_sec (default 300 s)
  - State (current_replicas, current_max_num_seqs, last_*_apply) advances ONLY on ok=True.
  - No destructive ops: only 'kubectl scale' and 'kubectl patch'.
  - tune_mode gates which paths run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from controller.config import ControllerConfig
from controller.kubectl_exec import KubectlCommandRunner, KubectlExecConfig
from controller.types import AppliedAction, PolicyDecision

LOG = logging.getLogger("kube-ai.actuator")


@dataclass(slots=True)
class ActuatorState:
    current_replicas: int
    current_max_num_seqs: int


class K8sActuator:
    def __init__(self, cfg: ControllerConfig) -> None:
        self.cfg = cfg
        init_replicas = max(cfg.min_replicas, min(cfg.max_replicas, 1))
        init_seqs = max(cfg.min_max_num_seqs, min(cfg.max_max_num_seqs, cfg.min_max_num_seqs))
        self.state = ActuatorState(
            current_replicas=init_replicas,
            current_max_num_seqs=init_seqs,
        )
        self.last_replica_apply: datetime = datetime.min.replace(tzinfo=timezone.utc)
        self.last_param_apply: datetime = datetime.min.replace(tzinfo=timezone.utc)
        self.runner = KubectlCommandRunner(
            KubectlExecConfig(
                mode=cfg.exec_mode,
                context=cfg.context,
                namespace=cfg.vllm_namespace,
                ssh_host=cfg.ssh_host,
                ssh_user=cfg.ssh_user,
                ssh_key_file=cfg.ssh_key_file,
                docker_container=cfg.docker_container,
            )
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scale_cmd(self, n: int) -> str:
        dep = self.cfg.vllm_deployment
        ns = self.cfg.vllm_namespace
        return f"scale deployment {dep} --namespace {ns} --replicas={n}"

    def _patch_cmd(self, max_num_seqs: int) -> str:
        dep = self.cfg.vllm_deployment
        ns = self.cfg.vllm_namespace
        patch = json.dumps(
            {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": dep,
                                    "args": [f"--max-num-seqs={max_num_seqs}"],
                                }
                            ]
                        }
                    }
                }
            }
        )
        return f"patch deployment {dep} --namespace {ns} --type=strategic -p {json.dumps(patch)}"

    def _run_or_dry(self, cmd: str, dry_run: bool) -> tuple[bool, str]:
        if dry_run:
            return True, f"DRY_RUN {cmd}"
        ok, out = self.runner.run(cmd, check=False)
        return ok, out

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def apply(self, decision: PolicyDecision) -> AppliedAction:
        """Apply a PolicyDecision to the cluster, respecting tune_mode and cooldowns."""
        now = datetime.now(timezone.utc)
        mode = self.cfg.tune_mode.lower()

        old_replicas = self.state.current_replicas
        old_max_num_seqs = self.state.current_max_num_seqs
        command_log: list[str] = []
        replica_changed = False
        param_changed = False

        # --- Path A: Replicas ---
        if mode in ("replicas", "both"):
            replica_cooldown_ok = (
                now - self.last_replica_apply >= timedelta(seconds=self.cfg.cooldown_sec)
            )
            if not replica_cooldown_ok:
                command_log.append(
                    f"replicas: cooldown ({self.cfg.cooldown_sec}s not elapsed), skip"
                )
            else:
                new_replicas = max(
                    self.cfg.min_replicas,
                    min(self.cfg.max_replicas, decision.target_replicas),
                )
                cmd = self._scale_cmd(new_replicas)
                ok, out = self._run_or_dry(cmd, self.cfg.dry_run)
                log_entry = f"{'DRY_RUN' if self.cfg.dry_run else ('OK' if ok else 'ERR')} {cmd}"
                if out and not self.cfg.dry_run:
                    log_entry += f" :: {out}"
                command_log.append(log_entry)

                if ok:
                    if new_replicas != old_replicas:
                        replica_changed = True
                    self.state.current_replicas = new_replicas
                    self.last_replica_apply = now
                else:
                    LOG.warning("scale command failed cmd=%s out=%s", cmd, out)

        # --- Path B: Params (max_num_seqs) ---
        if mode in ("params", "both"):
            param_cooldown_ok = (
                now - self.last_param_apply >= timedelta(seconds=self.cfg.param_cooldown_sec)
            )
            if not param_cooldown_ok:
                command_log.append(
                    f"params: cooldown ({self.cfg.param_cooldown_sec}s not elapsed), skip"
                )
            else:
                new_max_num_seqs = max(
                    self.cfg.min_max_num_seqs,
                    min(self.cfg.max_max_num_seqs, decision.target_max_num_seqs),
                )
                cmd = self._patch_cmd(new_max_num_seqs)
                ok, out = self._run_or_dry(cmd, self.cfg.dry_run)
                log_entry = f"{'DRY_RUN' if self.cfg.dry_run else ('OK' if ok else 'ERR')} {cmd}"
                if out and not self.cfg.dry_run:
                    log_entry += f" :: {out}"
                command_log.append(log_entry)

                if ok:
                    if new_max_num_seqs != old_max_num_seqs:
                        param_changed = True
                    self.state.current_max_num_seqs = new_max_num_seqs
                    self.last_param_apply = now
                else:
                    LOG.warning("patch command failed cmd=%s out=%s", cmd, out)

        return AppliedAction(
            changed=replica_changed or param_changed,
            old_replicas=old_replicas,
            new_replicas=self.state.current_replicas,
            old_max_num_seqs=old_max_num_seqs,
            new_max_num_seqs=self.state.current_max_num_seqs,
            command_log=command_log,
        )
