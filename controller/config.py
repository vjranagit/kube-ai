"""ControllerConfig — loaded from a YAML file then ENV overrides applied on top.

Both the YAML read and the os.getenv calls happen at **import time** via dataclass field
defaults.  Set environment variables before importing this module; there is no load_config().

YAML config file path: defaults to "config.yaml" at the repo root.  Override with
KUBE_AI_CONFIG env var.  If the file does not exist, all defaults come from ENV / hardcoded
values below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore[import]
except ImportError:
    yaml = None  # type: ignore[assignment]

_CONFIG_PATH = os.getenv("KUBE_AI_CONFIG", "config.yaml")


def _load_yaml(path: str) -> dict[str, Any]:
    """Load YAML config file; return empty dict if missing or yaml unavailable."""
    if yaml is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as f:
        data = yaml.safe_load(f) or {}
    return data


# Load once at import time.
_YAML: dict[str, Any] = _load_yaml(_CONFIG_PATH)


def _get(key: str, env_var: str, default: Any) -> Any:
    """Resolve: env override > yaml value > hardcoded default."""
    if (val := os.getenv(env_var)) is not None:
        return val
    if key in _YAML:
        return _YAML[key]
    return default


def _bool(key: str, env_var: str, default: bool) -> bool:
    raw = _get(key, env_var, str(default).lower())
    return str(raw).lower() == "true"


def _int(key: str, env_var: str, default: int) -> int:
    return int(_get(key, env_var, default))


def _float(key: str, env_var: str, default: float) -> float:
    return float(_get(key, env_var, default))


def _str(key: str, env_var: str, default: str) -> str:
    return str(_get(key, env_var, default))


@dataclass(slots=True)
class ControllerConfig:
    # --- Control loop ---
    tune_mode: str = field(default_factory=lambda: _str("tune_mode", "TUNE_MODE", "both"))
    interval_sec: int = field(default_factory=lambda: _int("interval_sec", "CONTROLLER_INTERVAL_SEC", 30))
    cooldown_sec: int = field(default_factory=lambda: _int("cooldown_sec", "CONTROLLER_COOLDOWN_SEC", 60))
    param_cooldown_sec: int = field(
        default_factory=lambda: _int("param_cooldown_sec", "CONTROLLER_PARAM_COOLDOWN_SEC", 300)
    )
    dry_run: bool = field(default_factory=lambda: _bool("dry_run", "CONTROLLER_DRY_RUN", True))

    # --- Replica bounds ---
    min_replicas: int = field(default_factory=lambda: _int("min_replicas", "MIN_REPLICAS", 1))
    max_replicas: int = field(default_factory=lambda: _int("max_replicas", "MAX_REPLICAS", 8))

    # --- max-num-seqs bounds ---
    min_max_num_seqs: int = field(
        default_factory=lambda: _int("min_max_num_seqs", "MIN_MAX_NUM_SEQS", 128)
    )
    max_max_num_seqs: int = field(
        default_factory=lambda: _int("max_max_num_seqs", "MAX_MAX_NUM_SEQS", 2048)
    )

    # --- Saturation thresholds ---
    pressure_high: float = field(default_factory=lambda: _float("pressure_high", "PRESSURE_HIGH", 0.75))
    pressure_low: float = field(default_factory=lambda: _float("pressure_low", "PRESSURE_LOW", 0.35))

    # --- Latency SLO ---
    ttft_slo_sec: float = field(default_factory=lambda: _float("ttft_slo_sec", "TTFT_SLO_SEC", 2.0))

    # --- Deployment target ---
    vllm_deployment: str = field(
        default_factory=lambda: _str("vllm_deployment", "VLLM_DEPLOYMENT", "vllm-server")
    )
    vllm_namespace: str = field(
        default_factory=lambda: _str("vllm_namespace", "VLLM_NAMESPACE", "default")
    )

    # --- vLLM metrics ---
    vllm_mode: str = field(default_factory=lambda: _str("vllm_mode", "VLLM_MODE", "mock"))
    vllm_metrics_url: str = field(
        default_factory=lambda: _str(
            "vllm_metrics_url", "VLLM_METRICS_URL", "http://localhost:8000/metrics"
        )
    )

    # --- kubectl execution ---
    exec_mode: str = field(default_factory=lambda: _str("exec_mode", "EXEC_MODE", "local"))
    context: str = field(default_factory=lambda: _str("context", "KUBECTL_CONTEXT", ""))
    namespace: str = field(
        default_factory=lambda: _str("namespace", "KUBECTL_NAMESPACE", "default")
    )
    ssh_host: str = field(default_factory=lambda: _str("ssh_host", "SSH_HOST", ""))
    ssh_user: str = field(default_factory=lambda: _str("ssh_user", "SSH_USER", ""))
    ssh_key_file: str = field(default_factory=lambda: _str("ssh_key_file", "SSH_KEY_FILE", ""))
    docker_container: str = field(
        default_factory=lambda: _str("docker_container", "DOCKER_CONTAINER", "")
    )

    # --- Tuner ---
    tuner_kind: str = field(default_factory=lambda: _str("tuner_kind", "TUNER_KIND", "aimd"))

    # --- Observability ---
    metrics_port: int = field(default_factory=lambda: _int("metrics_port", "METRICS_PORT", 9108))
