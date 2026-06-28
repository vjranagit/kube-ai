"""Tests for apps/api/main.py — FastAPI TestClient coverage of all endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# Import once; module-level cfg defaults to vllm_mode="mock" so no real cluster needed.
from apps.api.main import app, collector

_client = TestClient(app)


# ---------------------------------------------------------------------------
# GET /  — service info
# ---------------------------------------------------------------------------


def test_root_status_200() -> None:
    resp = _client.get("/")
    assert resp.status_code == 200


def test_root_returns_json() -> None:
    resp = _client.get("/")
    assert resp.headers["content-type"].startswith("application/json")


def test_root_service_name_is_kube_ai() -> None:
    resp = _client.get("/")
    assert resp.json()["service"] == "kube-ai"


def test_root_contains_health_key() -> None:
    resp = _client.get("/")
    data = resp.json()
    assert "health" in data


def test_root_contains_snapshot_key() -> None:
    resp = _client.get("/")
    data = resp.json()
    assert "snapshot" in data


def test_root_contains_metrics_key() -> None:
    resp = _client.get("/")
    data = resp.json()
    assert "metrics" in data


# ---------------------------------------------------------------------------
# GET /healthz  — liveness probe
# ---------------------------------------------------------------------------


def test_healthz_status_200() -> None:
    resp = _client.get("/healthz")
    assert resp.status_code == 200


def test_healthz_returns_ok() -> None:
    resp = _client.get("/healthz")
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /serving/snapshot
# ---------------------------------------------------------------------------


def test_serving_snapshot_status_200() -> None:
    resp = _client.get("/serving/snapshot")
    assert resp.status_code == 200


def test_serving_snapshot_contains_requests_waiting() -> None:
    resp = _client.get("/serving/snapshot")
    assert "requests_waiting" in resp.json()


def test_serving_snapshot_contains_kv_cache_usage_perc() -> None:
    resp = _client.get("/serving/snapshot")
    assert "kv_cache_usage_perc" in resp.json()


def test_serving_snapshot_metrics_available_true_in_mock_mode() -> None:
    resp = _client.get("/serving/snapshot")
    assert resp.json()["metrics_available"] is True


def test_serving_snapshot_mock_waiting_is_3() -> None:
    resp = _client.get("/serving/snapshot")
    assert resp.json()["requests_waiting"] == 3


def test_serving_snapshot_mock_running_is_5() -> None:
    resp = _client.get("/serving/snapshot")
    assert resp.json()["requests_running"] == 5


def test_serving_snapshot_contains_timestamp() -> None:
    resp = _client.get("/serving/snapshot")
    assert "timestamp" in resp.json()


# ---------------------------------------------------------------------------
# GET /deployment/status
# ---------------------------------------------------------------------------


def test_deployment_status_status_200() -> None:
    resp = _client.get("/deployment/status")
    assert resp.status_code == 200


def test_deployment_status_contains_deployment_key() -> None:
    resp = _client.get("/deployment/status")
    assert "deployment" in resp.json()


def test_deployment_status_contains_namespace_key() -> None:
    resp = _client.get("/deployment/status")
    assert "namespace" in resp.json()


def test_deployment_status_contains_desired_replicas() -> None:
    resp = _client.get("/deployment/status")
    assert "desired_replicas" in resp.json()


def test_deployment_status_contains_ready_replicas() -> None:
    resp = _client.get("/deployment/status")
    assert "ready_replicas" in resp.json()


def test_deployment_status_contains_metrics_available() -> None:
    resp = _client.get("/deployment/status")
    assert "metrics_available" in resp.json()


def test_deployment_status_deployment_name_matches_config() -> None:
    from controller.config import ControllerConfig

    default_deployment = ControllerConfig().vllm_deployment
    resp = _client.get("/deployment/status")
    assert resp.json()["deployment"] == default_deployment


# ---------------------------------------------------------------------------
# GET /metrics  — Prometheus text format
# ---------------------------------------------------------------------------


def test_metrics_status_200() -> None:
    resp = _client.get("/metrics")
    assert resp.status_code == 200


def test_metrics_content_type_is_prometheus() -> None:
    resp = _client.get("/metrics")
    assert "text/plain" in resp.headers["content-type"]


def test_metrics_body_contains_kube_ai_prefix() -> None:
    resp = _client.get("/metrics")
    assert "kube_ai_" in resp.text


def test_metrics_body_contains_saturation_gauge() -> None:
    resp = _client.get("/metrics")
    assert "kube_ai_saturation_score" in resp.text


def test_metrics_body_contains_requests_waiting_gauge() -> None:
    resp = _client.get("/metrics")
    assert "kube_ai_requests_waiting" in resp.text


# ---------------------------------------------------------------------------
# GET /ui  — static dashboard
# ---------------------------------------------------------------------------


def test_ui_index_returns_html() -> None:
    resp = _client.get("/ui/")
    # The dashboard is mounted; expect HTML content
    assert resp.status_code == 200
    assert "html" in resp.headers.get("content-type", "").lower()


def test_ui_index_html_contains_html_tag() -> None:
    resp = _client.get("/ui/")
    assert "<html" in resp.text.lower() or "<!doctype" in resp.text.lower()


# ---------------------------------------------------------------------------
# Monkeypatched snapshot — test that endpoint uses collector.snapshot()
# ---------------------------------------------------------------------------


def test_serving_snapshot_uses_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import datetime, timezone

    from controller.types import ServingSnapshot

    fake_snap = ServingSnapshot(
        timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
        desired_replicas=99,
        ready_replicas=99,
        available_replicas=99,
        requests_waiting=42,
        requests_running=7,
        requests_swapped=0,
        kv_cache_usage_perc=0.99,
        p95_ttft_sec=1.23,
        p50_ttft_sec=0.5,
        queue_pressure=0.5,
        cache_pressure=0.5,
        latency_pressure=0.0,
        current_max_num_seqs=512,
        metrics_available=True,
    )
    monkeypatch.setattr(collector, "snapshot", lambda: fake_snap)
    resp = _client.get("/serving/snapshot")
    data = resp.json()
    assert data["requests_waiting"] == 42
    assert data["desired_replicas"] == 99
