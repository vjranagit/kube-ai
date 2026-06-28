"""kube-ai FastAPI control-plane (separate process from the control loop).

Endpoints:
    GET /                   — service info
    GET /healthz            — liveness probe
    GET /serving/snapshot   — live ServingSnapshot (from K8sCollector)
    GET /deployment/status  — deployment replica counts
    GET /metrics            — Prometheus gauges (registered in this process)
    GET /ui                 — static dashboard (served from apps/dashboard/)

    # Web UI / new endpoints:
    GET  /api/state         — ServingSnapshot + kube_ai_* gauge values + config summary
    GET  /api/config        — current effective config (whitelisted fields only)
    POST /api/config        — validate & write config.yaml; update in-memory cfg
    POST /api/control/start — start control-loop subprocess
    POST /api/control/stop  — stop control-loop subprocess
    GET  /api/control/status — loop running state + pid + started_at

RBAC: all endpoints are currently unauthenticated.  Auth/authorization hooks belong in a
FastAPI dependency injected at the router or route level — see TODO comments below.

Note: this is a separate process from the control loop.  The loop's live Prometheus values are
on :9108; this API's /metrics endpoint shows the gauges registered in this process (zeroed
unless this process updates them independently).
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore[import]
except ImportError:
    yaml = None  # type: ignore[assignment]

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, field_validator
from starlette.responses import Response

from apps.api import control as _control
from controller import metrics as _metrics  # noqa: F401  # registers kube_ai_* gauges
from controller.collectors.k8s import K8sCollector
from controller.config import ControllerConfig

app = FastAPI(title="kube-ai", version="0.1.0")
cfg = ControllerConfig()
collector = K8sCollector(cfg)

_DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"
if _DASHBOARD_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_DASHBOARD_DIR), html=True), name="ui")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_CONFIG_WHITELIST: frozenset[str] = frozenset(
    {
        "cooldown_sec",
        "dry_run",
        "interval_sec",
        "max_max_num_seqs",
        "max_replicas",
        "min_max_num_seqs",
        "min_replicas",
        "param_cooldown_sec",
        "pressure_high",
        "pressure_low",
        "ttft_slo_sec",
        "tune_mode",
        "tuner_kind",
        "vllm_mode",
    }
)

_TUNE_MODES: frozenset[str] = frozenset({"replicas", "params", "both"})
_VLLM_MODES: frozenset[str] = frozenset({"mock", "real"})
_TUNER_KINDS: frozenset[str] = frozenset({"aimd", "rl"})


def _gauge_val(g: Any) -> float:
    """Read current value from a prometheus_client Gauge (uses internal _value.get())."""
    try:
        return float(g._value.get())  # type: ignore[attr-defined]
    except Exception:
        return 0.0


def _current_config_path() -> Path:
    return Path(os.getenv("KUBE_AI_CONFIG", "config.yaml"))


# ---------------------------------------------------------------------------
# Config update model (Pydantic v2)
# ---------------------------------------------------------------------------


class ConfigUpdate(BaseModel):
    """Accepted fields for POST /api/config.  Unknown keys are rejected (extra='forbid')."""

    model_config = ConfigDict(extra="forbid")

    tune_mode: str | None = None
    min_replicas: int | None = None
    max_replicas: int | None = None
    min_max_num_seqs: int | None = None
    max_max_num_seqs: int | None = None
    pressure_high: float | None = None
    pressure_low: float | None = None
    ttft_slo_sec: float | None = None
    cooldown_sec: int | None = None
    param_cooldown_sec: int | None = None
    interval_sec: int | None = None
    vllm_mode: str | None = None
    tuner_kind: str | None = None
    dry_run: bool | None = None

    @field_validator("tune_mode")
    @classmethod
    def check_tune_mode(cls, v: str | None) -> str | None:
        if v is not None and v not in _TUNE_MODES:
            raise ValueError(f"tune_mode must be one of {sorted(_TUNE_MODES)}")
        return v

    @field_validator("vllm_mode")
    @classmethod
    def check_vllm_mode(cls, v: str | None) -> str | None:
        if v is not None and v not in _VLLM_MODES:
            raise ValueError(f"vllm_mode must be one of {sorted(_VLLM_MODES)}")
        return v

    @field_validator("tuner_kind")
    @classmethod
    def check_tuner_kind(cls, v: str | None) -> str | None:
        if v is not None and v not in _TUNER_KINDS:
            raise ValueError(f"tuner_kind must be one of {sorted(_TUNER_KINDS)}")
        return v

    @field_validator("pressure_high", "pressure_low")
    @classmethod
    def check_pressure_range(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("pressure thresholds must be in [0.0, 1.0]")
        return v

    @field_validator("ttft_slo_sec")
    @classmethod
    def check_ttft_slo(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("ttft_slo_sec must be > 0")
        return v

    @field_validator("min_replicas", "max_replicas", "min_max_num_seqs", "max_max_num_seqs")
    @classmethod
    def check_non_negative(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("must be >= 0")
        return v

    @field_validator("cooldown_sec", "param_cooldown_sec", "interval_sec")
    @classmethod
    def check_positive_sec(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("must be >= 1")
        return v


# ---------------------------------------------------------------------------
# Existing endpoints — preserved unchanged
# ---------------------------------------------------------------------------


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "kube-ai",
        "dashboard": "/ui",
        "health": "/healthz",
        "snapshot": "/serving/snapshot",
        "deployment": "/deployment/status",
        "metrics": "/metrics",
        "state": "/api/state",
        "config": "/api/config",
        "control": "/api/control/status",
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/serving/snapshot")
def serving_snapshot() -> dict:
    snap = collector.snapshot()
    return asdict(snap)


@app.get("/deployment/status")
def deployment_status() -> dict:
    snap = collector.snapshot()
    return {
        "deployment": cfg.vllm_deployment,
        "namespace": cfg.vllm_namespace,
        "desired_replicas": snap.desired_replicas,
        "ready_replicas": snap.ready_replicas,
        "available_replicas": snap.available_replicas,
        "metrics_available": snap.metrics_available,
    }


@app.get("/metrics")
def metrics() -> Response:
    payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# New API endpoints
# ---------------------------------------------------------------------------


@app.get("/api/state")
def api_state() -> dict:
    """Live state: snapshot + kube_ai_* gauge values + config summary.

    RBAC TODO: add ``dependencies=[Depends(require_read_role)]`` to restrict access.
    """
    snap = collector.snapshot()
    snap_dict = asdict(snap)
    snap_dict["timestamp"] = str(snap_dict["timestamp"])

    gauges: dict[str, float] = {
        "kube_ai_saturation_score": _gauge_val(_metrics.SATURATION),
        "kube_ai_requests_waiting": _gauge_val(_metrics.REQUESTS_WAITING),
        "kube_ai_requests_running": _gauge_val(_metrics.REQUESTS_RUNNING),
        "kube_ai_kv_cache_usage_perc": _gauge_val(_metrics.KV_CACHE_USAGE),
        "kube_ai_target_replicas": _gauge_val(_metrics.TARGET_REPLICAS),
        "kube_ai_ready_replicas": _gauge_val(_metrics.READY_REPLICAS),
        "kube_ai_target_max_num_seqs": _gauge_val(_metrics.TARGET_MAX_NUM_SEQS),
        "kube_ai_p95_ttft_sec": _gauge_val(_metrics.P95_TTFT),
        "kube_ai_queue_pressure": _gauge_val(_metrics.QUEUE_PRESSURE),
        "kube_ai_cache_pressure": _gauge_val(_metrics.CACHE_PRESSURE),
        "kube_ai_latency_pressure": _gauge_val(_metrics.LATENCY_PRESSURE),
    }

    return {
        "snapshot": snap_dict,
        "gauges": gauges,
        "config_summary": {
            "tune_mode": cfg.tune_mode,
            "dry_run": cfg.dry_run,
            "pressure_high": cfg.pressure_high,
            "pressure_low": cfg.pressure_low,
            "ttft_slo_sec": cfg.ttft_slo_sec,
        },
    }


@app.get("/api/config")
def get_api_config() -> dict:
    """Return current effective config (whitelisted fields only).

    RBAC TODO: add read-role auth dependency here.
    """
    return {k: getattr(cfg, k) for k in sorted(_CONFIG_WHITELIST)}


@app.post("/api/config")
def post_api_config(update: ConfigUpdate) -> dict:
    """Validate and write config.yaml; update in-memory cfg.

    Only fields in _CONFIG_WHITELIST are accepted; unknown keys return 422 (pydantic).
    Semantic cross-field errors return 400.

    RBAC TODO: add write-role auth dependency here.
    """
    patch: dict[str, Any] = {
        k: v for k, v in update.model_dump().items() if v is not None
    }
    if not patch:
        raise HTTPException(status_code=400, detail="No fields provided")

    # Cross-field validation
    eff_min_r = patch.get("min_replicas", cfg.min_replicas)
    eff_max_r = patch.get("max_replicas", cfg.max_replicas)
    if eff_min_r > eff_max_r:
        raise HTTPException(status_code=400, detail="min_replicas must be <= max_replicas")

    eff_min_s = patch.get("min_max_num_seqs", cfg.min_max_num_seqs)
    eff_max_s = patch.get("max_max_num_seqs", cfg.max_max_num_seqs)
    if eff_min_s > eff_max_s:
        raise HTTPException(
            status_code=400, detail="min_max_num_seqs must be <= max_max_num_seqs"
        )

    eff_p_high = patch.get("pressure_high", cfg.pressure_high)
    eff_p_low = patch.get("pressure_low", cfg.pressure_low)
    if eff_p_low >= eff_p_high:
        raise HTTPException(status_code=400, detail="pressure_low must be < pressure_high")

    if yaml is None:
        raise HTTPException(status_code=500, detail="pyyaml not installed; cannot write config")

    config_path = _current_config_path()
    try:
        existing: dict[str, Any] = {}
        if config_path.exists():
            with config_path.open() as fh:
                existing = yaml.safe_load(fh) or {}
        existing.update(patch)
        with config_path.open("w") as fh:
            yaml.dump(existing, fh, default_flow_style=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write config: {exc}") from exc

    # Update in-memory cfg so subsequent /api/state reflects the change immediately.
    for k, v in patch.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return {"ok": True, "updated": patch, "config_path": str(config_path)}


@app.post("/api/control/start")
def control_start() -> dict:
    """Start the control-loop subprocess.

    RBAC TODO: add operator-role auth dependency here.
    """
    cp = _current_config_path()
    return _control.start_loop(str(cp) if cp.exists() else None)


@app.post("/api/control/stop")
def control_stop() -> dict:
    """Stop the control-loop subprocess.

    RBAC TODO: add operator-role auth dependency here.
    """
    return _control.stop_loop()


@app.get("/api/control/status")
def control_status() -> dict:
    """Return loop running state + pid + started_at.

    RBAC TODO: add read-role auth dependency here.
    """
    return _control.get_status()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=8080, reload=False)
