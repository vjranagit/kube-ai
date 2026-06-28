"""K8sCollector — snapshot() returns a ServingSnapshot.

In mock mode (vllm_mode=mock, the default), the collector exercises the real parsers against
a module-level static fixture so the loop runs fully offline.  Switching to real mode only
changes the *source* of the data; the parsers are identical.

Design rule: _parse_vllm_metrics() and _parse_deployment_json() are pure functions that take
text/JSON strings.  Never call subprocess or urllib inside them; callers handle the I/O.
"""

from __future__ import annotations

import json
import logging
import shlex
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from controller.config import ControllerConfig
from controller.kubectl_exec import KubectlCommandRunner, KubectlExecConfig
from controller.types import ServingSnapshot

LOG = logging.getLogger("kube-ai.collector")

# ---------------------------------------------------------------------------
# Static fixture for mock mode
# ---------------------------------------------------------------------------

# Minimal Prometheus text-format metrics blob that exercises all parsers.
_MOCK_VLLM_METRICS = """\
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="mistral"} 3.0
# HELP vllm:num_requests_running Number of requests currently being processed.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="mistral"} 5.0
# HELP vllm:num_requests_swapped Number of requests swapped to CPU.
# TYPE vllm:num_requests_swapped gauge
vllm:num_requests_swapped{model_name="mistral"} 0.0
# HELP vllm:kv_cache_usage_perc GPU KV-cache usage in percent.
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

# Minimal kubectl get deployment -o json output.
_MOCK_DEPLOYMENT_JSON = json.dumps(
    {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "vllm-server", "namespace": "default"},
        "spec": {"replicas": 2},
        "status": {
            "replicas": 2,
            "readyReplicas": 2,
            "availableReplicas": 2,
        },
    }
)


# ---------------------------------------------------------------------------
# Pure parsers (source-agnostic)
# ---------------------------------------------------------------------------


def _parse_vllm_metrics(text: str) -> dict[str, Any]:
    """Parse Prometheus text-format output from vLLM /metrics.

    Returns a dict with keys:
        num_requests_waiting, num_requests_running, num_requests_swapped,
        kv_cache_usage_perc, p95_ttft_sec, p50_ttft_sec
    Absent metrics default to 0.0.
    """
    result: dict[str, Any] = {
        "num_requests_waiting": 0.0,
        "num_requests_running": 0.0,
        "num_requests_swapped": 0.0,
        "kv_cache_usage_perc": 0.0,
        "p95_ttft_sec": 0.0,
        "p50_ttft_sec": 0.0,
    }

    ttft_buckets: list[tuple[float, float]] = []  # (le, cumulative_count)
    ttft_count = 0.0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Split metric name (with labels) from value
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        metric_expr, value_str = parts
        try:
            value = float(value_str)
        except ValueError:
            continue

        # Strip labels: metric_name{...} → metric_name
        metric_name = metric_expr.split("{")[0]

        if metric_name == "vllm:num_requests_waiting":
            result["num_requests_waiting"] = value
        elif metric_name == "vllm:num_requests_running":
            result["num_requests_running"] = value
        elif metric_name == "vllm:num_requests_swapped":
            result["num_requests_swapped"] = value
        elif metric_name == "vllm:kv_cache_usage_perc":
            result["kv_cache_usage_perc"] = value
        elif metric_name == "vllm:time_to_first_token_seconds_bucket":
            # Extract le label value
            le_val = _extract_label(metric_expr, "le")
            if le_val is not None:
                try:
                    le = float(le_val) if le_val != "+Inf" else float("inf")
                    ttft_buckets.append((le, value))
                except ValueError:
                    pass
        elif metric_name == "vllm:time_to_first_token_seconds_count":
            ttft_count = value

    if ttft_buckets and ttft_count > 0:
        ttft_buckets.sort(key=lambda t: t[0])
        result["p95_ttft_sec"] = _percentile_from_buckets(ttft_buckets, ttft_count, 0.95)
        result["p50_ttft_sec"] = _percentile_from_buckets(ttft_buckets, ttft_count, 0.50)

    return result


def _extract_label(metric_expr: str, label: str) -> str | None:
    """Extract a label value from a Prometheus metric expression like name{k="v",le="+Inf"}."""
    start = metric_expr.find("{")
    end = metric_expr.rfind("}")
    if start == -1 or end == -1:
        return None
    labels_str = metric_expr[start + 1 : end]
    for pair in labels_str.split(","):
        pair = pair.strip()
        if pair.startswith(label + "="):
            return pair[len(label) + 1 :].strip('"')
    return None


def _percentile_from_buckets(
    buckets: list[tuple[float, float]], total_count: float, quantile: float
) -> float:
    """Approximate a quantile from Prometheus histogram buckets (linear interpolation)."""
    target = quantile * total_count
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if count >= target:
            if count == prev_count:
                return le
            # Linear interpolation within bucket
            frac = (target - prev_count) / (count - prev_count)
            lower = prev_le
            upper = le if le != float("inf") else prev_le * 2 or 10.0
            return lower + frac * (upper - lower)
        prev_le, prev_count = le, count
    # All counts below target: return last finite bucket upper bound
    for le, _ in reversed(buckets):
        if le != float("inf"):
            return le
    return 0.0


def _parse_deployment_json(js: str) -> dict[str, int]:
    """Parse kubectl get deployment -o json output.

    Returns dict with desired_replicas, ready_replicas, available_replicas.
    """
    try:
        data = json.loads(js)
    except json.JSONDecodeError:
        return {"desired_replicas": 1, "ready_replicas": 0, "available_replicas": 0}

    spec = data.get("spec", {})
    status = data.get("status", {})
    return {
        "desired_replicas": int(spec.get("replicas", 1)),
        "ready_replicas": int(status.get("readyReplicas", 0)),
        "available_replicas": int(status.get("availableReplicas", 0)),
    }


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class K8sCollector:
    """Collects a ServingSnapshot from vLLM metrics + kubectl deployment state."""

    def __init__(self, cfg: ControllerConfig) -> None:
        self.cfg = cfg
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
    # Data acquisition
    # ------------------------------------------------------------------

    def _fetch_vllm_metrics(self) -> str:
        """Fetch raw Prometheus text from vLLM /metrics endpoint."""
        try:
            req = urllib.request.Request(self.cfg.vllm_metrics_url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            LOG.warning("vLLM metrics fetch failed url=%s err=%s", self.cfg.vllm_metrics_url, exc)
            return ""

    def _fetch_deployment_json(self) -> str:
        """Run kubectl get deployment -o json; return raw JSON string.

        Deployment name and namespace are shlex-quoted to prevent shell injection (C2).
        """
        dep_raw = self.cfg.vllm_deployment
        ns_raw = self.cfg.vllm_namespace
        dep = shlex.quote(dep_raw)
        ns = shlex.quote(ns_raw)
        cmd = f"get deployment {dep} --namespace {ns} -o json"
        ok, out = self.runner.run(cmd, check=False)
        if not ok:
            LOG.warning("kubectl get deployment failed dep=%s err=%s", dep_raw, out)
            return "{}"
        return out

    @staticmethod
    def _parse_max_num_seqs(deployment_json: str) -> int | None:
        """Extract --max-num-seqs=N from Deployment container args.

        Returns None if the arg is absent (caller should fall back to config default).
        """
        try:
            data = json.loads(deployment_json)
        except json.JSONDecodeError:
            return None
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
                        return int(arg.split("=", 1)[1])
                    except ValueError:
                        pass
        return None

    # ------------------------------------------------------------------
    # Snapshot assembly
    # ------------------------------------------------------------------

    def snapshot(self) -> ServingSnapshot:
        if self.cfg.vllm_mode == "mock":
            metrics_text = _MOCK_VLLM_METRICS
            deployment_json = _MOCK_DEPLOYMENT_JSON
            metrics_available = True
        else:
            metrics_text = self._fetch_vllm_metrics()
            deployment_json = self._fetch_deployment_json()
            metrics_available = bool(metrics_text)

        vllm = _parse_vllm_metrics(metrics_text)
        dep = _parse_deployment_json(deployment_json)

        # Prefer the live --max-num-seqs arg from Deployment container args; fall back
        # to cfg.min_max_num_seqs so mock/offline mode still works.
        if self.cfg.vllm_mode == "mock":
            current_max_num_seqs = self.cfg.min_max_num_seqs
        else:
            parsed_seqs = self._parse_max_num_seqs(deployment_json)
            current_max_num_seqs = parsed_seqs if parsed_seqs is not None else self.cfg.min_max_num_seqs

        return ServingSnapshot(
            timestamp=datetime.now(timezone.utc),
            desired_replicas=dep["desired_replicas"],
            ready_replicas=dep["ready_replicas"],
            available_replicas=dep["available_replicas"],
            requests_waiting=int(vllm["num_requests_waiting"]),
            requests_running=int(vllm["num_requests_running"]),
            requests_swapped=int(vllm["num_requests_swapped"]),
            kv_cache_usage_perc=float(vllm["kv_cache_usage_perc"]),
            p95_ttft_sec=float(vllm["p95_ttft_sec"]),
            p50_ttft_sec=float(vllm["p50_ttft_sec"]),
            # Pressure sub-scores computed below; policy engine will recompute from snap
            # but we embed them for logging convenience.
            queue_pressure=0.0,
            cache_pressure=0.0,
            latency_pressure=0.0,
            current_max_num_seqs=current_max_num_seqs,
            metrics_available=metrics_available,
        )

    def info_json(self) -> str:
        snap = self.snapshot()
        return json.dumps(asdict(snap), default=str)
