"""kube-ai FastAPI control-plane (separate process from the control loop).

Endpoints:
    GET /                 — service info
    GET /healthz          — liveness probe
    GET /serving/snapshot — live ServingSnapshot (from K8sCollector)
    GET /deployment/status — deployment replica counts
    GET /metrics          — Prometheus gauges (registered in this process)
    GET /ui               — static dashboard

Note: this is a separate process from the control loop.  The loop's live Prometheus values are
on :9108; this API's /metrics endpoint shows the gauges registered in this process (zeroed
unless this process updates them independently).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from controller import metrics as _metrics  # noqa: F401  # registers kube_ai_* gauges
from controller.collectors.k8s import K8sCollector
from controller.config import ControllerConfig

app = FastAPI(title="kube-ai", version="0.1.0")
cfg = ControllerConfig()
collector = K8sCollector(cfg)

_DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"
if _DASHBOARD_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_DASHBOARD_DIR), html=True), name="ui")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "kube-ai",
        "dashboard": "/ui",
        "health": "/healthz",
        "snapshot": "/serving/snapshot",
        "deployment": "/deployment/status",
        "metrics": "/metrics",
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=8080, reload=False)
