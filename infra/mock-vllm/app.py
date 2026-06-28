#!/usr/bin/env python3
"""mock-vllm: tiny Flask server simulating a vLLM /metrics endpoint.

Synthetic load model:
  - Default: sawtooth wave cycling 0→1→0 over SAWTOOTH_PERIOD seconds.
  - POST /admin/set-load {"level": 0..1}  pins load to a fixed level.
  - GET  /metrics  returns Prometheus text with real vLLM metric names.
  - GET  /healthz  returns "ok".
  - GET  /v1/completions  returns a stub 200 JSON.

Replica-aware capacity:
  - Set MOCK_REPLICAS env var (or update via POST /admin/set-replicas {"replicas": N}).
  - Higher replicas → lower waiting (load spread across more capacity).
"""

from __future__ import annotations

import math
import os
import threading
import time

from flask import Flask, Response, jsonify, request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = os.environ.get("MODEL_NAME", "mistral")
SAWTOOTH_PERIOD = float(os.environ.get("SAWTOOTH_PERIOD", "60"))  # seconds per full cycle

# ---------------------------------------------------------------------------
# Shared state (guarded by _lock)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_load_level: float = 0.0       # current synthetic load [0..1]
_manual_override: bool = False  # True after POST /admin/set-load
_replicas: int = max(1, int(os.environ.get("MOCK_REPLICAS", "1")))

_start_time: float = time.time()
_request_success_total: float = 0.0
_generation_tokens_total: float = 0.0

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Background sawtooth thread
# ---------------------------------------------------------------------------


def _sawtooth_worker() -> None:
    """Advance sawtooth load model every second unless manually overridden."""
    global _load_level
    t0 = time.monotonic()
    half = SAWTOOTH_PERIOD / 2.0
    while True:
        time.sleep(1.0)
        with _lock:
            if _manual_override:
                continue
            elapsed = (time.monotonic() - t0) % SAWTOOTH_PERIOD
            _load_level = elapsed / half if elapsed < half else 1.0 - (elapsed - half) / half


_bg = threading.Thread(target=_sawtooth_worker, daemon=True)
_bg.start()

# ---------------------------------------------------------------------------
# Metrics builder
# ---------------------------------------------------------------------------


def _build_metrics() -> str:
    global _request_success_total, _generation_tokens_total

    with _lock:
        level = _load_level
        reps = _replicas

    # --- Gauge metrics ---
    # waiting: offered load divided by replica capacity; drops as replicas increase
    waiting = max(0.0, level * 80.0 / reps)
    # running: scales with replicas (more capacity → more concurrent handling)
    running = max(0.0, level * reps * 8.0)
    # swapped: only under extreme load (>85%)
    swapped = max(0.0, (level - 0.85) * 20.0) if level > 0.85 else 0.0
    kv_cache = min(0.99, level * 0.85)

    # --- TTFT histogram (exponential distribution approximation) ---
    # mean TTFT: 0.1 s at no load → 6.0 s at full load
    mean_ttft = 0.1 + level * 5.9
    count = max(1, round(running * 2) or 1)

    ttft_les = [0.5, 1.0, 2.0, 4.0, 8.0, math.inf]

    def _ecdf(x: float, mean: float) -> float:
        """Exponential CDF: 1 - exp(-x/mean)."""
        return 1.0 - math.exp(-x / mean) if mean > 0 else 1.0

    ttft_buckets = [round(_ecdf(le, mean_ttft) * count) if not math.isinf(le) else count
                    for le in ttft_les]
    ttft_sum = mean_ttft * count

    # --- e2e latency histogram ---
    mean_e2e = mean_ttft * 3.0
    e2e_les = [1.0, 2.0, 4.0, 8.0, 16.0, math.inf]
    e2e_buckets = [round(_ecdf(le, mean_e2e) * count) if not math.isinf(le) else count
                   for le in e2e_les]
    e2e_sum = mean_e2e * count

    # --- Monotonic counters (time-weighted) ---
    elapsed = time.time() - _start_time
    rst = int(elapsed * level * 2.0)
    gtt = int(elapsed * level * 100.0)

    m = MODEL
    lines: list[str] = []

    def _gauge(name: str, help_text: str, value: float) -> None:
        lines.append(f"# HELP vllm:{name} {help_text}")
        lines.append(f"# TYPE vllm:{name} gauge")
        lines.append(f'vllm:{name}{{model_name="{m}"}} {value:.4f}')

    _gauge("num_requests_waiting", "Number of requests waiting to be processed.", waiting)
    _gauge("num_requests_running", "Number of requests currently being processed.", running)
    _gauge("num_requests_swapped", "Number of requests swapped to CPU.", swapped)
    _gauge("kv_cache_usage_perc", "GPU KV-cache usage in percent.", kv_cache)

    # TTFT histogram
    lines.append("# HELP vllm:time_to_first_token_seconds Histogram of time to first token.")
    lines.append("# TYPE vllm:time_to_first_token_seconds histogram")
    for le, cnt in zip(ttft_les, ttft_buckets):
        le_str = "+Inf" if math.isinf(le) else str(le)
        lines.append(f'vllm:time_to_first_token_seconds_bucket{{model_name="{m}",le="{le_str}"}} {cnt}')
    lines.append(f'vllm:time_to_first_token_seconds_sum{{model_name="{m}"}} {ttft_sum:.3f}')
    lines.append(f'vllm:time_to_first_token_seconds_count{{model_name="{m}"}} {count}')

    # e2e latency histogram
    lines.append("# HELP vllm:e2e_request_latency_seconds Histogram of end-to-end request latency.")
    lines.append("# TYPE vllm:e2e_request_latency_seconds histogram")
    for le, cnt in zip(e2e_les, e2e_buckets):
        le_str = "+Inf" if math.isinf(le) else str(le)
        lines.append(f'vllm:e2e_request_latency_seconds_bucket{{model_name="{m}",le="{le_str}"}} {cnt}')
    lines.append(f'vllm:e2e_request_latency_seconds_sum{{model_name="{m}"}} {e2e_sum:.3f}')
    lines.append(f'vllm:e2e_request_latency_seconds_count{{model_name="{m}"}} {count}')

    # Counters
    lines.append("# HELP vllm:request_success_total Total successful requests.")
    lines.append("# TYPE vllm:request_success_total counter")
    lines.append(f'vllm:request_success_total{{model_name="{m}"}} {rst}')
    lines.append("# HELP vllm:generation_tokens_total Total generated tokens.")
    lines.append("# TYPE vllm:generation_tokens_total counter")
    lines.append(f'vllm:generation_tokens_total{{model_name="{m}"}} {gtt}')

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/metrics")
def metrics() -> Response:
    return Response(_build_metrics(), mimetype="text/plain; version=0.0.4; charset=utf-8")


@app.get("/healthz")
def healthz() -> Response:
    return Response("ok\n", mimetype="text/plain")


@app.get("/v1/completions")
@app.post("/v1/completions")
def completions_stub() -> Response:
    """Stub endpoint so the host controller or load tests can hit a real-ish URL."""
    body = {
        "id": "cmpl-stub",
        "object": "text_completion",
        "model": MODEL,
        "choices": [{"text": "stub response", "index": 0, "finish_reason": "stop"}],
    }
    return jsonify(body)


@app.post("/admin/set-load")
def set_load() -> Response:
    """Override synthetic load level. Body: {"level": 0.0..1.0}."""
    global _load_level, _manual_override
    data = request.get_json(force=True, silent=True) or {}
    level = float(data.get("level", 0.5))
    level = max(0.0, min(1.0, level))
    with _lock:
        _load_level = level
        _manual_override = True
    return jsonify({"ok": True, "level": level})


@app.post("/admin/set-replicas")
def set_replicas() -> Response:
    """Update replica count (mirrors what the actuator sets via kubectl scale)."""
    global _replicas
    data = request.get_json(force=True, silent=True) or {}
    reps = max(1, int(data.get("replicas", 1)))
    with _lock:
        _replicas = reps
    return jsonify({"ok": True, "replicas": reps})


@app.post("/admin/reset")
def reset() -> Response:
    """Resume sawtooth (cancel manual override)."""
    global _manual_override
    with _lock:
        _manual_override = False
    return jsonify({"ok": True, "mode": "sawtooth"})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
