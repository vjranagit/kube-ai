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
import shlex
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
        # Sync initial state from the live Deployment so that scale-in AIMD
        # correctly halves from the actual running replica count, not from 1.
        self._sync_initial_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_initial_state(self) -> None:
        """Read current Deployment state from the cluster and sync both replicas and
        max_num_seqs.  (H3: was replica-only; now also syncs max_num_seqs.)

        Skips silently if the cluster is unreachable (state keeps the safe defaults).
        Deployment name and namespace are shlex-quoted to prevent shell injection (C2).
        """
        dep = shlex.quote(self.cfg.vllm_deployment)
        ns = shlex.quote(self.cfg.vllm_namespace)
        cmd = f"get deployment {dep} --namespace {ns} -o json"
        try:
            ok, out = self.runner.run(cmd, check=False)
            if not ok:
                return
            data = json.loads(out)
            # Sync replica count from spec.replicas
            live_replicas = data.get("spec", {}).get("replicas")
            if live_replicas is not None:
                try:
                    clamped = max(
                        self.cfg.min_replicas,
                        min(self.cfg.max_replicas, int(live_replicas)),
                    )
                    self.state.current_replicas = clamped
                    LOG.info("actuator initial replicas synced from cluster: %d", clamped)
                except (TypeError, ValueError):
                    pass
            # Sync max_num_seqs from container args (H3)
            containers = (
                data.get("spec", {})
                .get("template", {})
                .get("spec", {})
                .get("containers", [])
            )
            for container in containers:
                for arg in container.get("args", []):
                    if arg.startswith("--max-num-seqs="):
                        try:
                            parsed = int(arg.split("=", 1)[1])
                            clamped_seqs = max(
                                self.cfg.min_max_num_seqs,
                                min(self.cfg.max_max_num_seqs, parsed),
                            )
                            self.state.current_max_num_seqs = clamped_seqs
                            LOG.info(
                                "actuator initial max_num_seqs synced from cluster: %d",
                                clamped_seqs,
                            )
                        except (TypeError, ValueError):
                            pass
        except Exception as exc:  # noqa: BLE001
            LOG.warning("failed to sync initial state from cluster: %s", exc)

    def _scale_cmd(self, n: int) -> str:
        """Build an injection-safe kubectl scale command (C2)."""
        dep = shlex.quote(self.cfg.vllm_deployment)
        ns = shlex.quote(self.cfg.vllm_namespace)
        return f"scale deployment {dep} --namespace {ns} --replicas={n}"

    def _get_deployment_json(self) -> str:
        """Fetch current deployment JSON from the cluster. Returns '{}' on failure."""
        dep = shlex.quote(self.cfg.vllm_deployment)
        ns = shlex.quote(self.cfg.vllm_namespace)
        cmd = f"get deployment {dep} --namespace {ns} -o json"
        try:
            ok, out = self.runner.run(cmd, check=False)
            return out if (ok and out) else "{}"
        except Exception:  # noqa: BLE001
            return "{}"

    @staticmethod
    def _parse_container_args(deployment_json: str, container_name: str) -> list[str]:
        """Extract args for the named container from a deployment JSON string.

        Falls back to the first container if the named one is not found.
        Returns [] on any parse failure.
        """
        if not deployment_json:
            return []
        try:
            data = json.loads(deployment_json)
        except (json.JSONDecodeError, TypeError, ValueError):
            return []
        containers = (
            data.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        )
        for container in containers:
            if container.get("name") == container_name:
                return list(container.get("args", []))
        if containers:
            return list(containers[0].get("args", []))
        return []

    def _patch_cmd(self, max_num_seqs: int, current_args: list[str]) -> str:
        """Build an injection-safe kubectl patch command that preserves all container
        args except --max-num-seqs=N (C2 + C4).

        Args:
            max_num_seqs: The target value to set for --max-num-seqs.
            current_args: The live container args list; only --max-num-seqs is
                replaced or appended; all other flags are preserved.
        """
        dep = shlex.quote(self.cfg.vllm_deployment)
        ns = shlex.quote(self.cfg.vllm_namespace)
        new_arg = f"--max-num-seqs={max_num_seqs}"
        # Replace existing --max-num-seqs=* in-place; append if absent
        replaced = False
        updated_args: list[str] = []
        for arg in current_args:
            if arg.startswith("--max-num-seqs="):
                updated_args.append(new_arg)
                replaced = True
            else:
                updated_args.append(arg)
        if not replaced:
            updated_args.append(new_arg)
        patch = json.dumps(
            {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": self.cfg.vllm_deployment,
                                    "args": updated_args,
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
                if new_replicas == old_replicas:
                    # H4: idempotency — skip kubectl when nothing would change
                    command_log.append(f"replicas: no change ({new_replicas}), skip")
                else:
                    cmd = self._scale_cmd(new_replicas)
                    ok, out = self._run_or_dry(cmd, self.cfg.dry_run)
                    log_entry = f"{'DRY_RUN' if self.cfg.dry_run else ('OK' if ok else 'ERR')} {cmd}"
                    if out and not self.cfg.dry_run:
                        log_entry += f" :: {out}"
                    command_log.append(log_entry)

                    if ok:
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
                if new_max_num_seqs == old_max_num_seqs:
                    # H4: idempotency — skip kubectl when nothing would change
                    command_log.append(f"params: no change ({new_max_num_seqs}), skip")
                else:
                    # C4: fetch live container args to preserve all flags except --max-num-seqs
                    if self.cfg.dry_run:
                        live_args: list[str] = []
                    else:
                        live_args = self._parse_container_args(
                            self._get_deployment_json(), self.cfg.vllm_deployment
                        )
                    cmd = self._patch_cmd(new_max_num_seqs, live_args)
                    ok, out = self._run_or_dry(cmd, self.cfg.dry_run)
                    log_entry = f"{'DRY_RUN' if self.cfg.dry_run else ('OK' if ok else 'ERR')} {cmd}"
                    if out and not self.cfg.dry_run:
                        log_entry += f" :: {out}"
                    command_log.append(log_entry)

                    if ok:
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
