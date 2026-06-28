"""Tests for collectors/k8s.py — parsers, snapshot assembly, edge cases."""
from __future__ import annotations

import json

import pytest

from controller.collectors.k8s import (
    K8sCollector,
    _parse_deployment_json,
    _parse_vllm_metrics,
)
from controller.tests.conftest import (
    MOCK_DEPLOYMENT_JSON,
    VLLM_METRICS_EMPTY,
    VLLM_METRICS_NO_HISTOGRAM,
    VLLM_METRICS_TYPICAL,
    VLLM_METRICS_WITH_SWAP,
    MockMetricsServer,
    make_cfg,
)


# ---------------------------------------------------------------------------
# _parse_vllm_metrics — gauge values
# ---------------------------------------------------------------------------


def test_parse_vllm_metrics_waiting_count() -> None:
    result = _parse_vllm_metrics(VLLM_METRICS_TYPICAL)
    assert result["num_requests_waiting"] == pytest.approx(3.0)


def test_parse_vllm_metrics_running_count() -> None:
    result = _parse_vllm_metrics(VLLM_METRICS_TYPICAL)
    assert result["num_requests_running"] == pytest.approx(5.0)


def test_parse_vllm_metrics_swapped_count() -> None:
    result = _parse_vllm_metrics(VLLM_METRICS_TYPICAL)
    assert result["num_requests_swapped"] == pytest.approx(0.0)


def test_parse_vllm_metrics_kv_cache_usage() -> None:
    result = _parse_vllm_metrics(VLLM_METRICS_TYPICAL)
    assert result["kv_cache_usage_perc"] == pytest.approx(0.42)


def test_parse_vllm_metrics_p50_ttft() -> None:
    # Bucket (0.5,2)→(1.0,5): target=5, frac=1.0 → p50=1.0
    result = _parse_vllm_metrics(VLLM_METRICS_TYPICAL)
    assert result["p50_ttft_sec"] == pytest.approx(1.0)


def test_parse_vllm_metrics_p95_ttft() -> None:
    # target=9.5, last finite bucket upper=4, next=+Inf → interp to 6.0
    result = _parse_vllm_metrics(VLLM_METRICS_TYPICAL)
    assert result["p95_ttft_sec"] == pytest.approx(6.0)


def test_parse_vllm_metrics_swapped_positive() -> None:
    result = _parse_vllm_metrics(VLLM_METRICS_WITH_SWAP)
    assert result["num_requests_swapped"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# _parse_vllm_metrics — empty / missing metrics
# ---------------------------------------------------------------------------


def test_parse_vllm_metrics_empty_string_defaults_to_zero() -> None:
    result = _parse_vllm_metrics(VLLM_METRICS_EMPTY)
    assert result["num_requests_waiting"] == pytest.approx(0.0)
    assert result["num_requests_running"] == pytest.approx(0.0)
    assert result["kv_cache_usage_perc"] == pytest.approx(0.0)


def test_parse_vllm_metrics_no_histogram_p95_is_zero() -> None:
    result = _parse_vllm_metrics(VLLM_METRICS_NO_HISTOGRAM)
    assert result["p95_ttft_sec"] == pytest.approx(0.0)


def test_parse_vllm_metrics_no_histogram_p50_is_zero() -> None:
    result = _parse_vllm_metrics(VLLM_METRICS_NO_HISTOGRAM)
    assert result["p50_ttft_sec"] == pytest.approx(0.0)


def test_parse_vllm_metrics_comment_lines_ignored() -> None:
    text = "# HELP something ignored\n# TYPE something gauge\nvllm:num_requests_waiting 7.0\n"
    result = _parse_vllm_metrics(text)
    assert result["num_requests_waiting"] == pytest.approx(7.0)


def test_parse_vllm_metrics_malformed_value_skipped() -> None:
    text = "vllm:num_requests_waiting{} notanumber\nvllm:num_requests_running{} 4.0\n"
    result = _parse_vllm_metrics(text)
    assert result["num_requests_waiting"] == pytest.approx(0.0)
    assert result["num_requests_running"] == pytest.approx(4.0)


def test_parse_vllm_metrics_missing_label_braces_ignored() -> None:
    text = "vllm:num_requests_waiting 2.0\n"
    result = _parse_vllm_metrics(text)
    assert result["num_requests_waiting"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _parse_deployment_json — typical
# ---------------------------------------------------------------------------


def test_parse_deployment_json_desired_replicas() -> None:
    result = _parse_deployment_json(MOCK_DEPLOYMENT_JSON)
    assert result["desired_replicas"] == 2


def test_parse_deployment_json_ready_replicas() -> None:
    result = _parse_deployment_json(MOCK_DEPLOYMENT_JSON)
    assert result["ready_replicas"] == 2


def test_parse_deployment_json_available_replicas() -> None:
    result = _parse_deployment_json(MOCK_DEPLOYMENT_JSON)
    assert result["available_replicas"] == 2


def test_parse_deployment_json_missing_ready_defaults_to_zero() -> None:
    js = json.dumps({"spec": {"replicas": 3}, "status": {"availableReplicas": 2}})
    result = _parse_deployment_json(js)
    assert result["ready_replicas"] == 0


def test_parse_deployment_json_missing_spec_replicas_defaults_to_one() -> None:
    js = json.dumps({"spec": {}, "status": {"readyReplicas": 0}})
    result = _parse_deployment_json(js)
    assert result["desired_replicas"] == 1


def test_parse_deployment_json_malformed_json_returns_defaults() -> None:
    result = _parse_deployment_json("{invalid json}")
    assert result["desired_replicas"] == 1
    assert result["ready_replicas"] == 0
    assert result["available_replicas"] == 0


def test_parse_deployment_json_empty_string_returns_defaults() -> None:
    result = _parse_deployment_json("")
    assert result["desired_replicas"] == 1


# ---------------------------------------------------------------------------
# snapshot() in mock mode — uses static fixture, no I/O
# ---------------------------------------------------------------------------


def test_snapshot_mock_mode_returns_serving_snapshot() -> None:
    cfg = make_cfg(vllm_mode="mock")
    collector = K8sCollector(cfg)
    snap = collector.snapshot()
    assert snap.requests_waiting == 3
    assert snap.requests_running == 5


def test_snapshot_mock_mode_metrics_available_true() -> None:
    cfg = make_cfg(vllm_mode="mock")
    collector = K8sCollector(cfg)
    snap = collector.snapshot()
    assert snap.metrics_available is True


def test_snapshot_mock_mode_kv_cache_correct() -> None:
    cfg = make_cfg(vllm_mode="mock")
    collector = K8sCollector(cfg)
    snap = collector.snapshot()
    assert snap.kv_cache_usage_perc == pytest.approx(0.42)


def test_snapshot_mock_mode_p95_ttft_correct() -> None:
    cfg = make_cfg(vllm_mode="mock")
    collector = K8sCollector(cfg)
    snap = collector.snapshot()
    assert snap.p95_ttft_sec == pytest.approx(6.0)


def test_snapshot_mock_mode_desired_replicas() -> None:
    cfg = make_cfg(vllm_mode="mock")
    collector = K8sCollector(cfg)
    snap = collector.snapshot()
    assert snap.desired_replicas == 2


# ---------------------------------------------------------------------------
# snapshot() in real mode — monkeypatched runner + MockMetricsServer
# ---------------------------------------------------------------------------


def test_snapshot_real_mode_assembles_from_live_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    server = MockMetricsServer(VLLM_METRICS_TYPICAL)
    try:
        cfg = make_cfg(vllm_mode="real", vllm_metrics_url=server.url)
        collector = K8sCollector(cfg)
        monkeypatch.setattr(collector.runner, "run", lambda cmd, check=True: (True, MOCK_DEPLOYMENT_JSON))
        snap = collector.snapshot()
        assert snap.requests_waiting == 3
        assert snap.requests_running == 5
        assert snap.metrics_available is True
    finally:
        server.close()


def test_snapshot_real_mode_deployment_json_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    server = MockMetricsServer(VLLM_METRICS_TYPICAL)
    try:
        cfg = make_cfg(vllm_mode="real", vllm_metrics_url=server.url)
        collector = K8sCollector(cfg)
        monkeypatch.setattr(collector.runner, "run", lambda cmd, check=True: (True, MOCK_DEPLOYMENT_JSON))
        snap = collector.snapshot()
        assert snap.desired_replicas == 2
        assert snap.ready_replicas == 2
        assert snap.available_replicas == 2
    finally:
        server.close()


def test_snapshot_real_mode_empty_metrics_sets_metrics_available_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = MockMetricsServer(VLLM_METRICS_EMPTY)
    try:
        cfg = make_cfg(vllm_mode="real", vllm_metrics_url=server.url)
        collector = K8sCollector(cfg)
        monkeypatch.setattr(collector.runner, "run", lambda cmd, check=True: (True, MOCK_DEPLOYMENT_JSON))
        snap = collector.snapshot()
        assert snap.metrics_available is False
    finally:
        server.close()


def test_snapshot_real_mode_unreachable_endpoint_metrics_available_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_cfg(vllm_mode="real", vllm_metrics_url="http://127.0.0.1:1/metrics")
    collector = K8sCollector(cfg)
    monkeypatch.setattr(collector.runner, "run", lambda cmd, check=True: (True, MOCK_DEPLOYMENT_JSON))
    snap = collector.snapshot()
    assert snap.metrics_available is False
    assert snap.requests_waiting == 0
    assert snap.requests_running == 0


def test_snapshot_real_mode_malformed_deployment_json_uses_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = MockMetricsServer(VLLM_METRICS_TYPICAL)
    try:
        cfg = make_cfg(vllm_mode="real", vllm_metrics_url=server.url)
        collector = K8sCollector(cfg)
        monkeypatch.setattr(
            collector.runner, "run", lambda cmd, check=True: (True, "{not valid json}")
        )
        snap = collector.snapshot()
        assert snap.desired_replicas == 1
        assert snap.ready_replicas == 0
    finally:
        server.close()


def test_snapshot_real_mode_runner_failure_falls_back_to_empty_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = MockMetricsServer(VLLM_METRICS_TYPICAL)
    try:
        cfg = make_cfg(vllm_mode="real", vllm_metrics_url=server.url)
        collector = K8sCollector(cfg)
        monkeypatch.setattr(
            collector.runner, "run", lambda cmd, check=True: (False, "permission denied")
        )
        snap = collector.snapshot()
        # Should not raise; falls back to default desired_replicas=1
        assert snap.desired_replicas == 1
    finally:
        server.close()


# ---------------------------------------------------------------------------
# snapshot() timestamp is recent
# ---------------------------------------------------------------------------


def test_snapshot_timestamp_is_timezone_aware() -> None:

    cfg = make_cfg(vllm_mode="mock")
    collector = K8sCollector(cfg)
    snap = collector.snapshot()
    assert snap.timestamp.tzinfo is not None


def test_snapshot_pressure_scores_initialised_to_zero() -> None:
    """Collector sets sub-scores to 0.0; policy engine computes them."""
    cfg = make_cfg(vllm_mode="mock")
    collector = K8sCollector(cfg)
    snap = collector.snapshot()
    assert snap.queue_pressure == pytest.approx(0.0)
    assert snap.cache_pressure == pytest.approx(0.0)
    assert snap.latency_pressure == pytest.approx(0.0)
