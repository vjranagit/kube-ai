"""KubectlCommandRunner — single subprocess choke point for ALL kubectl calls.

Every kubectl interaction in this project MUST go through KubectlCommandRunner.run().
No other module may call subprocess directly.

Modes:
  local  — run kubectl on the host directly
  ssh    — run kubectl on a remote host via SSH
  docker — run kubectl inside a docker container via docker exec
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass


@dataclass(slots=True)
class KubectlExecConfig:
    mode: str               # local | ssh | docker
    context: str = ""       # kubectl --context (blank = current-context)
    namespace: str = "default"
    ssh_host: str = ""
    ssh_user: str = ""
    ssh_key_file: str = ""
    docker_container: str = ""


class KubectlCommandRunner:
    """Wraps kubectl CLI calls with a timeout; never raises on command failure.

    Returns (ok: bool, output: str).  ok is True iff returncode == 0.
    """

    _TIMEOUT_SEC = 30

    def __init__(self, cfg: KubectlExecConfig) -> None:
        self.cfg = cfg

    def _kubectl_prefix(self) -> str:
        """Build the 'kubectl [--context X] [--namespace N]' prefix."""
        parts = ["kubectl"]
        if self.cfg.context:
            parts += ["--context", shlex.quote(self.cfg.context)]
        if self.cfg.namespace:
            parts += ["--namespace", shlex.quote(self.cfg.namespace)]
        return " ".join(parts)

    def _build(self, command: str) -> str:
        """Wrap a kubectl subcommand in the appropriate exec mode."""
        mode = self.cfg.mode.strip().lower()
        kubectl_cmd = f"{self._kubectl_prefix()} {command}"

        if mode == "local":
            return kubectl_cmd

        if mode == "ssh":
            if not self.cfg.ssh_host:
                raise ValueError("SSH_HOST is required when exec_mode=ssh")
            identity = f"-i {shlex.quote(self.cfg.ssh_key_file)} " if self.cfg.ssh_key_file else ""
            strict = "-o StrictHostKeyChecking=accept-new -o BatchMode=yes"
            user_host = (
                f"{self.cfg.ssh_user}@{self.cfg.ssh_host}"
                if self.cfg.ssh_user
                else self.cfg.ssh_host
            )
            return f"ssh {identity}{strict} {shlex.quote(user_host)} {shlex.quote(kubectl_cmd)}"

        if mode == "docker":
            if not self.cfg.docker_container:
                raise ValueError("DOCKER_CONTAINER is required when exec_mode=docker")
            return (
                f"docker exec {shlex.quote(self.cfg.docker_container)} "
                f"sh -c {shlex.quote(kubectl_cmd)}"
            )

        raise ValueError(f"unsupported exec_mode: {self.cfg.mode!r}")

    def run(self, command: str, check: bool = True) -> tuple[bool, str]:
        """Run a kubectl subcommand; return (ok, output).  Never raises."""
        wrapped = self._build(command)
        try:
            proc = subprocess.run(
                wrapped,
                shell=True,
                check=check,
                capture_output=True,
                text=True,
                timeout=self._TIMEOUT_SEC,
            )
            return proc.returncode == 0, proc.stdout.strip()
        except subprocess.CalledProcessError as exc:
            msg = (exc.stderr or exc.stdout or str(exc)).strip()
            return False, msg
        except subprocess.TimeoutExpired:
            return False, f"kubectl timed out after {self._TIMEOUT_SEC}s: {command}"
