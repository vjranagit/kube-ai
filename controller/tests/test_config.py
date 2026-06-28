"""Tests for config.py — defaults and env overrides via subprocess."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

PYTHON = sys.executable

# Env vars that ControllerConfig reads — strip them from the parent so tests are hermetic.
_CONFIG_ENV_KEYS = [
    "KUBE_AI_CONFIG",
    "TUNE_MODE",
    "CONTROLLER_INTERVAL_SEC",
    "CONTROLLER_COOLDOWN_SEC",
    "CONTROLLER_PARAM_COOLDOWN_SEC",
    "CONTROLLER_DRY_RUN",
    "MIN_REPLICAS",
    "MAX_REPLICAS",
    "MIN_MAX_NUM_SEQS",
    "MAX_MAX_NUM_SEQS",
    "PRESSURE_HIGH",
    "PRESSURE_LOW",
    "TTFT_SLO_SEC",
    "VLLM_DEPLOYMENT",
    "VLLM_NAMESPACE",
    "VLLM_MODE",
    "VLLM_METRICS_URL",
    "EXEC_MODE",
    "KUBECTL_CONTEXT",
    "KUBECTL_NAMESPACE",
    "SSH_HOST",
    "SSH_USER",
    "SSH_KEY_FILE",
    "DOCKER_CONTAINER",
    "TUNER_KIND",
    "METRICS_PORT",
]


def run_cfg_expr(expr: str, env: dict[str, str] | None = None) -> str:
    """Run expr in a subprocess that imports ControllerConfig and prints a field."""
    base_env = {k: v for k, v in os.environ.items()}
    for key in _CONFIG_ENV_KEYS:
        base_env.pop(key, None)
    if env:
        base_env.update(env)
    code = (
        "from controller.config import ControllerConfig; "
        f"cfg = ControllerConfig(); print({expr})"
    )
    result = subprocess.run(
        [PYTHON, "-c", code],
        capture_output=True,
        text=True,
        env=base_env,
        timeout=10,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_default_tune_mode_is_both() -> None:
    assert run_cfg_expr("cfg.tune_mode") == "both"


def test_default_interval_sec_is_30() -> None:
    assert run_cfg_expr("cfg.interval_sec") == "30"


def test_default_cooldown_sec_is_60() -> None:
    assert run_cfg_expr("cfg.cooldown_sec") == "60"


def test_default_param_cooldown_sec_is_300() -> None:
    assert run_cfg_expr("cfg.param_cooldown_sec") == "300"


def test_default_dry_run_is_true() -> None:
    assert run_cfg_expr("cfg.dry_run") == "True"


def test_default_min_replicas_is_1() -> None:
    assert run_cfg_expr("cfg.min_replicas") == "1"


def test_default_max_replicas_is_8() -> None:
    assert run_cfg_expr("cfg.max_replicas") == "8"


def test_default_min_max_num_seqs_is_128() -> None:
    assert run_cfg_expr("cfg.min_max_num_seqs") == "128"


def test_default_max_max_num_seqs_is_2048() -> None:
    assert run_cfg_expr("cfg.max_max_num_seqs") == "2048"


def test_default_pressure_high_is_0_75() -> None:
    val = float(run_cfg_expr("cfg.pressure_high"))
    assert val == pytest.approx(0.75)


def test_default_pressure_low_is_0_35() -> None:
    val = float(run_cfg_expr("cfg.pressure_low"))
    assert val == pytest.approx(0.35)


def test_default_ttft_slo_sec_is_2_0() -> None:
    val = float(run_cfg_expr("cfg.ttft_slo_sec"))
    assert val == pytest.approx(2.0)


def test_default_vllm_deployment_is_vllm_server() -> None:
    assert run_cfg_expr("cfg.vllm_deployment") == "vllm-server"


def test_default_vllm_namespace_is_default() -> None:
    assert run_cfg_expr("cfg.vllm_namespace") == "default"


def test_default_vllm_mode_is_mock() -> None:
    assert run_cfg_expr("cfg.vllm_mode") == "mock"


def test_default_exec_mode_is_local() -> None:
    assert run_cfg_expr("cfg.exec_mode") == "local"


def test_default_context_is_empty() -> None:
    assert run_cfg_expr("cfg.context") == ""


def test_default_ssh_host_is_empty() -> None:
    assert run_cfg_expr("cfg.ssh_host") == ""


def test_default_tuner_kind_is_aimd() -> None:
    assert run_cfg_expr("cfg.tuner_kind") == "aimd"


def test_default_metrics_port_is_9108() -> None:
    assert run_cfg_expr("cfg.metrics_port") == "9108"


# ---------------------------------------------------------------------------
# Env overrides win over hardcoded defaults
# ---------------------------------------------------------------------------


def test_env_override_tune_mode_replicas() -> None:
    assert run_cfg_expr("cfg.tune_mode", {"TUNE_MODE": "replicas"}) == "replicas"


def test_env_override_interval_sec() -> None:
    assert run_cfg_expr("cfg.interval_sec", {"CONTROLLER_INTERVAL_SEC": "15"}) == "15"


def test_env_override_cooldown_sec() -> None:
    assert run_cfg_expr("cfg.cooldown_sec", {"CONTROLLER_COOLDOWN_SEC": "120"}) == "120"


def test_env_override_dry_run_false() -> None:
    assert run_cfg_expr("cfg.dry_run", {"CONTROLLER_DRY_RUN": "false"}) == "False"


def test_env_override_dry_run_true_uppercase() -> None:
    assert run_cfg_expr("cfg.dry_run", {"CONTROLLER_DRY_RUN": "TRUE"}) == "True"


def test_env_override_min_replicas() -> None:
    assert run_cfg_expr("cfg.min_replicas", {"MIN_REPLICAS": "2"}) == "2"


def test_env_override_max_replicas() -> None:
    assert run_cfg_expr("cfg.max_replicas", {"MAX_REPLICAS": "16"}) == "16"


def test_env_override_pressure_high() -> None:
    val = float(run_cfg_expr("cfg.pressure_high", {"PRESSURE_HIGH": "0.9"}))
    assert val == pytest.approx(0.9)


def test_env_override_pressure_low() -> None:
    val = float(run_cfg_expr("cfg.pressure_low", {"PRESSURE_LOW": "0.2"}))
    assert val == pytest.approx(0.2)


def test_env_override_vllm_mode_real() -> None:
    assert run_cfg_expr("cfg.vllm_mode", {"VLLM_MODE": "real"}) == "real"


def test_env_override_exec_mode_ssh() -> None:
    assert run_cfg_expr("cfg.exec_mode", {"EXEC_MODE": "ssh"}) == "ssh"


def test_env_override_tuner_kind_rl() -> None:
    assert run_cfg_expr("cfg.tuner_kind", {"TUNER_KIND": "rl"}) == "rl"


def test_env_override_metrics_port() -> None:
    assert run_cfg_expr("cfg.metrics_port", {"METRICS_PORT": "9200"}) == "9200"


# ---------------------------------------------------------------------------
# TUNE_MODE valid values round-trip
# ---------------------------------------------------------------------------


def test_tune_mode_params_roundtrip() -> None:
    assert run_cfg_expr("cfg.tune_mode", {"TUNE_MODE": "params"}) == "params"


def test_tune_mode_both_roundtrip() -> None:
    assert run_cfg_expr("cfg.tune_mode", {"TUNE_MODE": "both"}) == "both"


# ---------------------------------------------------------------------------
# Bounds sanity: defaults satisfy min < max
# ---------------------------------------------------------------------------


def test_default_min_replicas_less_than_max_replicas() -> None:
    mn = int(run_cfg_expr("cfg.min_replicas"))
    mx = int(run_cfg_expr("cfg.max_replicas"))
    assert mn < mx


def test_default_pressure_low_less_than_pressure_high() -> None:
    low = float(run_cfg_expr("cfg.pressure_low"))
    high = float(run_cfg_expr("cfg.pressure_high"))
    assert low < high


def test_default_min_max_num_seqs_less_than_max_max_num_seqs() -> None:
    mn = int(run_cfg_expr("cfg.min_max_num_seqs"))
    mx = int(run_cfg_expr("cfg.max_max_num_seqs"))
    assert mn < mx
