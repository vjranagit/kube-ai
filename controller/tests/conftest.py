"""Shared fixtures for the kube-ai test suite."""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

import controller.config as _cc_module
from controller.config import ControllerConfig

# ---------------------------------------------------------------------------
# Test isolation: clear the module-level _YAML cache NOW, before any test
# module is imported.  apps/api/main.py runs  cfg = ControllerConfig()  at
# import time; if _YAML still holds live values from a repo-root config.yaml,
# those tests will get wrong defaults.  Resetting here (at conftest import
# time) ensures all subsequent imports see an empty cache.
# ---------------------------------------------------------------------------
_cc_module._YAML = {}


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------


def make_cfg(**overrides: Any) -> ControllerConfig:
    """Build a ControllerConfig with safe test defaults (no real config.yaml needed)."""
    params: dict[str, Any] = dict(
        tune_mode="both",
        interval_sec=30,
        cooldown_sec=60,
        param_cooldown_sec=300,
        dry_run=True,
        min_replicas=1,
        max_replicas=8,
        min_max_num_seqs=128,
        max_max_num_seqs=2048,
        pressure_high=0.75,
        pressure_low=0.35,
        ttft_slo_sec=2.0,
        vllm_deployment="vllm-server",
        vllm_namespace="default",
        vllm_mode="mock",
        vllm_metrics_url="http://localhost:8000/metrics",
        exec_mode="local",
        context="",
        namespace="default",
        ssh_host="",
        ssh_user="",
        ssh_key_file="",
        docker_container="",
        tuner_kind="aimd",
        metrics_port=9108,
        # M1: RL fields omitted previously caused config.yaml bleed-through in E2E env.
        rl_qtable_path="/nonexistent/qtable.json",
        rl_alpha=0.1,
        rl_gamma=0.9,
        rl_epsilon=0.1,
        rl_train_episodes=300,
    )
    params.update(overrides)
    return ControllerConfig(**params)


@pytest.fixture
def default_cfg() -> ControllerConfig:
    """ControllerConfig with safe test defaults (no env side-effects)."""
    return make_cfg()


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Autouse per-test: keep controller.config._YAML empty for every test.

    Prevents any on-disk config.yaml from leaking live values into
    ControllerConfig() calls that omit explicit parameters.
    """
    monkeypatch.setattr(_cc_module, "_YAML", {})


# ---------------------------------------------------------------------------
# FakeRunner — single seam so no test shells out to a real cluster
# ---------------------------------------------------------------------------


class FakeRunner:
    """Drop-in for KubectlCommandRunner.run returning canned (ok, out) pairs.

    Responses cycle through the list; the last entry is repeated once exhausted.
    All calls are recorded in ``calls``.
    """

    def __init__(self, responses: list[tuple[bool, str]] | None = None) -> None:
        self.responses: list[tuple[bool, str]] = list(responses or [(True, "")])
        self._call_index = 0
        self.calls: list[str] = []

    def run(self, command: str, check: bool = True) -> tuple[bool, str]:  # noqa: ARG002
        self.calls.append(command)
        idx = min(self._call_index, len(self.responses) - 1)
        self._call_index += 1
        return self.responses[idx]


def patch_runner(monkeypatch: pytest.MonkeyPatch, obj: Any, fake: FakeRunner) -> None:
    """Monkeypatch obj.runner.run with fake.run."""
    monkeypatch.setattr(obj.runner, "run", fake.run)


# ---------------------------------------------------------------------------
# Canned Prometheus metric bodies
# ---------------------------------------------------------------------------

VLLM_METRICS_TYPICAL = """\
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="mistral"} 3.0
# HELP vllm:num_requests_running Number of requests currently being processed.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="mistral"} 5.0
# HELP vllm:num_requests_swapped Number of requests swapped to CPU.
# TYPE vllm:num_requests_swapped gauge
vllm:num_requests_swapped{model_name="mistral"} 0.0
# HELP vllm:kv_cache_usage_perc GPU KV-cache usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="mistral"} 0.42
# HELP vllm:time_to_first_token_seconds Histogram of time to first token.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="mistral",le="0.5"} 2.0
vllm:time_to_first_token_seconds_bucket{model_name="mistral",le="1.0"} 5.0
vllm:time_to_first_token_seconds_bucket{model_name="mistral",le="2.0"} 8.0
vllm:time_to_first_token_seconds_bucket{model_name="mistral",le="4.0"} 9.0
vllm:time_to_first_token_seconds_bucket{model_name="mistral",le="+Inf"} 10.0
vllm:time_to_first_token_seconds_sum{model_name="mistral"} 18.5
vllm:time_to_first_token_seconds_count{model_name="mistral"} 10.0
"""

VLLM_METRICS_EMPTY = ""

VLLM_METRICS_NO_HISTOGRAM = """\
vllm:num_requests_waiting{model_name="m"} 1.0
vllm:num_requests_running{model_name="m"} 2.0
vllm:num_requests_swapped{model_name="m"} 0.0
vllm:kv_cache_usage_perc{model_name="m"} 0.5
"""

VLLM_METRICS_WITH_SWAP = """\
vllm:num_requests_waiting{model_name="m"} 1.0
vllm:num_requests_running{model_name="m"} 2.0
vllm:num_requests_swapped{model_name="m"} 3.0
vllm:kv_cache_usage_perc{model_name="m"} 0.5
"""

MOCK_DEPLOYMENT_JSON = """{
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {"name": "vllm-server", "namespace": "default"},
    "spec": {"replicas": 2},
    "status": {
        "replicas": 2,
        "readyReplicas": 2,
        "availableReplicas": 2
    }
}"""


# ---------------------------------------------------------------------------
# MockMetricsServer — serves canned vLLM Prometheus text on localhost
# ---------------------------------------------------------------------------


class _MetricsHandler(BaseHTTPRequestHandler):
    """Simple handler that serves a static body for any GET."""

    body: bytes = b""

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.end_headers()
        self.wfile.write(self.__class__.body)

    def log_message(self, fmt: str, *args: object) -> None:  # silence access log
        pass


class MockMetricsServer:
    """Thread-backed HTTP server for canned Prometheus metric text.

    Usage::

        server = MockMetricsServer(VLLM_METRICS_TYPICAL)
        # server.url  — http://127.0.0.1:<ephemeral-port>/metrics
        server.close()
    """

    def __init__(self, body: str) -> None:
        handler_cls = type("_H", (_MetricsHandler,), {"body": body.encode()})
        self._server = HTTPServer(("127.0.0.1", 0), handler_cls)
        self.port: int = self._server.server_address[1]
        self.url: str = f"http://127.0.0.1:{self.port}/metrics"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()


@pytest.fixture
def mock_metrics_server() -> MockMetricsServer:
    """Fixture: MockMetricsServer serving VLLM_METRICS_TYPICAL, auto-closed after test."""
    server = MockMetricsServer(VLLM_METRICS_TYPICAL)
    yield server  # type: ignore[misc]
    server.close()
