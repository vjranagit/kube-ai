"""Tests for Web UI endpoints added in apps/api/main.py.

Covers: GET /api/state, GET /api/config, POST /api/config (valid + invalid),
        GET /api/control/status, GET /ui (returns HTML).

These tests use FastAPI TestClient (no real cluster; vllm_mode=mock).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app

_client = TestClient(app)


# ---------------------------------------------------------------------------
# GET /api/state
# ---------------------------------------------------------------------------


def test_api_state_returns_200() -> None:
    resp = _client.get("/api/state")
    assert resp.status_code == 200


def test_api_state_has_snapshot_key() -> None:
    data = _client.get("/api/state").json()
    assert "snapshot" in data


def test_api_state_snapshot_has_requests_waiting() -> None:
    data = _client.get("/api/state").json()
    assert "requests_waiting" in data["snapshot"]


def test_api_state_snapshot_has_kv_cache_usage_perc() -> None:
    data = _client.get("/api/state").json()
    assert "kv_cache_usage_perc" in data["snapshot"]


def test_api_state_has_gauges_key() -> None:
    data = _client.get("/api/state").json()
    assert "gauges" in data


def test_api_state_gauges_has_saturation_score() -> None:
    data = _client.get("/api/state").json()
    assert "kube_ai_saturation_score" in data["gauges"]


def test_api_state_gauges_has_target_replicas() -> None:
    data = _client.get("/api/state").json()
    assert "kube_ai_target_replicas" in data["gauges"]


def test_api_state_has_config_summary() -> None:
    data = _client.get("/api/state").json()
    assert "config_summary" in data


def test_api_state_config_summary_has_tune_mode() -> None:
    data = _client.get("/api/state").json()
    assert "tune_mode" in data["config_summary"]


def test_api_state_config_summary_has_pressure_thresholds() -> None:
    data = _client.get("/api/state").json()
    cs = data["config_summary"]
    assert "pressure_high" in cs
    assert "pressure_low" in cs


def test_api_state_config_summary_has_dry_run() -> None:
    data = _client.get("/api/state").json()
    assert "dry_run" in data["config_summary"]


def test_api_state_mock_snapshot_waiting_is_3() -> None:
    data = _client.get("/api/state").json()
    # mock mode fixture always returns 3 waiting requests
    assert data["snapshot"]["requests_waiting"] == 3


# ---------------------------------------------------------------------------
# GET /api/config
# ---------------------------------------------------------------------------


def test_api_config_get_returns_200() -> None:
    resp = _client.get("/api/config")
    assert resp.status_code == 200


def test_api_config_get_has_tune_mode() -> None:
    data = _client.get("/api/config").json()
    assert "tune_mode" in data


def test_api_config_get_has_dry_run() -> None:
    data = _client.get("/api/config").json()
    assert "dry_run" in data


def test_api_config_get_has_pressure_fields() -> None:
    data = _client.get("/api/config").json()
    assert "pressure_high" in data
    assert "pressure_low" in data


def test_api_config_get_has_all_whitelisted_fields() -> None:
    data = _client.get("/api/config").json()
    expected = {
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
    assert expected.issubset(set(data.keys()))


# ---------------------------------------------------------------------------
# POST /api/config — valid updates
# ---------------------------------------------------------------------------


def test_api_config_post_valid_tune_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import apps.api.main as _m

    monkeypatch.setattr(_m, "_current_config_path", lambda: tmp_path / "config.yaml")
    resp = _client.post("/api/config", json={"tune_mode": "both"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_api_config_post_valid_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import apps.api.main as _m

    monkeypatch.setattr(_m, "_current_config_path", lambda: tmp_path / "config.yaml")
    resp = _client.post("/api/config", json={"dry_run": True})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_api_config_post_updated_field_in_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import apps.api.main as _m

    monkeypatch.setattr(_m, "_current_config_path", lambda: tmp_path / "config.yaml")
    resp = _client.post("/api/config", json={"tune_mode": "replicas"})
    body = resp.json()
    assert body["ok"] is True
    assert body["updated"]["tune_mode"] == "replicas"


# ---------------------------------------------------------------------------
# POST /api/config — invalid / rejected inputs
# ---------------------------------------------------------------------------


def test_api_config_post_invalid_tune_mode_returns_422() -> None:
    resp = _client.post("/api/config", json={"tune_mode": "garbage"})
    assert resp.status_code == 422


def test_api_config_post_invalid_vllm_mode_returns_422() -> None:
    resp = _client.post("/api/config", json={"vllm_mode": "kubernetes"})
    assert resp.status_code == 422


def test_api_config_post_pressure_out_of_range_returns_422() -> None:
    resp = _client.post("/api/config", json={"pressure_high": 1.5})
    assert resp.status_code == 422


def test_api_config_post_negative_pressure_returns_422() -> None:
    resp = _client.post("/api/config", json={"pressure_low": -0.1})
    assert resp.status_code == 422


def test_api_config_post_unknown_key_rejected_422() -> None:
    resp = _client.post("/api/config", json={"unknown_field": "value", "tune_mode": "both"})
    assert resp.status_code == 422


def test_api_config_post_pressure_low_gte_high_returns_400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import apps.api.main as _m

    monkeypatch.setattr(_m, "_current_config_path", lambda: tmp_path / "config.yaml")
    resp = _client.post("/api/config", json={"pressure_low": 0.9, "pressure_high": 0.5})
    assert resp.status_code == 400


def test_api_config_post_empty_body_returns_400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import apps.api.main as _m

    monkeypatch.setattr(_m, "_current_config_path", lambda: tmp_path / "config.yaml")
    resp = _client.post("/api/config", json={})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/control/status
# ---------------------------------------------------------------------------


def test_api_control_status_returns_200() -> None:
    resp = _client.get("/api/control/status")
    assert resp.status_code == 200


def test_api_control_status_has_running_field() -> None:
    data = _client.get("/api/control/status").json()
    assert "running" in data


def test_api_control_status_has_pid_field() -> None:
    data = _client.get("/api/control/status").json()
    assert "pid" in data


def test_api_control_status_not_running_in_test_env() -> None:
    # Loop subprocess is never started in unit-test context.
    data = _client.get("/api/control/status").json()
    assert data["running"] is False


# ---------------------------------------------------------------------------
# GET /ui — static dashboard
# ---------------------------------------------------------------------------


def test_ui_index_returns_200() -> None:
    resp = _client.get("/ui/")
    assert resp.status_code == 200


def test_ui_index_content_type_is_html() -> None:
    resp = _client.get("/ui/")
    ct = resp.headers.get("content-type", "")
    assert "html" in ct.lower()


def test_ui_index_contains_kube_ai() -> None:
    resp = _client.get("/ui/")
    assert "kube-ai" in resp.text.lower()


def test_ui_index_has_chart_js_script_tag() -> None:
    resp = _client.get("/ui/")
    assert "chart.js" in resp.text.lower()
