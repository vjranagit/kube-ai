#!/usr/bin/env python3
"""serving-shim — turn a real OpenAI-compatible backend (Ollama/llama.cpp/TGI/vLLM) into a
source of vLLM-style Prometheus metrics, computed from REAL traffic.

Why this exists: kube-ai's collector reads vLLM's native `vllm:*` metrics. When the real model
is served by an engine that runs on the available hardware (e.g. Ollama on a Pascal GPU, where
vLLM itself cannot run), this shim sits in front of it and measures genuine serving pressure —
in-flight vs queued requests against a capacity limit, plus real latency — and exports it as
`vllm:*` metrics. So the kube-ai loop is driven by a REAL model under REAL load, unmodified.

Stdlib only. Run:
    OLLAMA_URL=http://HOST:11500 MODEL=qwen2.5:0.5b MAX_CONCURRENCY=4 PORT=8000 \
        python3 infra/serving-shim/app.py

Endpoints:
    POST /v1/completions   -> proxied to <OLLAMA_URL>/v1/completions (real generation)
    GET  /metrics          -> Prometheus text with vllm:* names derived from live traffic
    GET  /health           -> 200 ok
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11500").rstrip("/")
MODEL = os.environ.get("MODEL", "qwen2.5:0.5b")
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "4"))
PORT = int(os.environ.get("PORT", "8000"))

_LAT_BUCKETS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0]

_lock = threading.Lock()
_sem = threading.BoundedSemaphore(MAX_CONCURRENCY)
_running = 0          # requests actively forwarded to the backend
_waiting = 0          # requests received but waiting for a capacity slot
_success_total = 0
_gen_tokens_total = 0
_ttft_sum = 0.0
_ttft_count = 0
_ttft_bucket = {b: 0 for b in _LAT_BUCKETS}
_e2e_sum = 0.0
_e2e_count = 0
_e2e_bucket = {b: 0 for b in _LAT_BUCKETS}


def _observe(latency: float, sum_ref: str) -> None:
    global _ttft_sum, _ttft_count, _e2e_sum, _e2e_count
    with _lock:
        if sum_ref == "ttft":
            _ttft_sum += latency
            _ttft_count += 1
            for b in _LAT_BUCKETS:
                if latency <= b:
                    _ttft_bucket[b] += 1
        else:
            _e2e_sum += latency
            _e2e_count += 1
            for b in _LAT_BUCKETS:
                if latency <= b:
                    _e2e_bucket[b] += 1


def _forward(body: bytes) -> tuple[int, bytes, int]:
    """Forward a completion request to the real backend; return (status, body, completion_tokens)."""
    req = urllib.request.Request(
        f"{OLLAMA_URL}/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
        toks = 0
        try:
            toks = int(json.loads(data).get("usage", {}).get("completion_tokens", 0))
        except Exception:
            pass
        return resp.status, data, toks


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # silence default logging
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, b'{"status":"ok"}')
        elif self.path == "/metrics":
            self._send(200, _render_metrics().encode(), "text/plain; version=0.0.4")
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self) -> None:
        if not self.path.startswith("/v1/completions"):
            self._send(404, b'{"error":"only /v1/completions"}')
            return
        global _running, _waiting, _success_total, _gen_tokens_total
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        # force the configured model
        try:
            payload = json.loads(raw or b"{}")
        except Exception:
            payload = {}
        payload.setdefault("model", MODEL)
        body = json.dumps(payload).encode()

        with _lock:
            _waiting += 1
        _sem.acquire()
        with _lock:
            _waiting -= 1
            _running += 1
        start = time.monotonic()
        try:
            status, data, toks = _forward(body)
            latency = time.monotonic() - start
            _observe(latency, "ttft")   # non-streaming: e2e used as TTFT proxy (documented)
            _observe(latency, "e2e")
            with _lock:
                if status == 200:
                    _success_total += 1
                    _gen_tokens_total += toks
            self._send(status, data)
        except Exception as exc:
            self._send(502, json.dumps({"error": str(exc)}).encode())
        finally:
            with _lock:
                _running -= 1
            _sem.release()


def _render_metrics() -> str:
    with _lock:
        running, waiting = _running, _waiting
        success, gen_toks = _success_total, _gen_tokens_total
        ttft_sum, ttft_count = _ttft_sum, _ttft_count
        e2e_sum, e2e_count = _e2e_sum, _e2e_count
        ttft_b = dict(_ttft_bucket)
        e2e_b = dict(_e2e_bucket)
    # kv_cache proxy: how full the serving capacity is right now (real backend busy-ness)
    kv = min(1.0, running / max(1, MAX_CONCURRENCY))
    lbl = f'model_name="{MODEL}"'
    lines = [
        "# HELP vllm:num_requests_running Requests currently executing on the backend.",
        "# TYPE vllm:num_requests_running gauge",
        f"vllm:num_requests_running{{{lbl}}} {running}",
        "# TYPE vllm:num_requests_waiting gauge",
        f"vllm:num_requests_waiting{{{lbl}}} {waiting}",
        "# TYPE vllm:num_requests_swapped gauge",
        f"vllm:num_requests_swapped{{{lbl}}} 0",
        "# TYPE vllm:kv_cache_usage_perc gauge",
        f"vllm:kv_cache_usage_perc{{{lbl}}} {kv:.4f}",
        "# TYPE vllm:request_success_total counter",
        f"vllm:request_success_total{{{lbl}}} {success}",
        "# TYPE vllm:generation_tokens_total counter",
        f"vllm:generation_tokens_total{{{lbl}}} {gen_toks}",
    ]
    for name, buckets, ssum, scount in (
        ("vllm:time_to_first_token_seconds", ttft_b, ttft_sum, ttft_count),
        ("vllm:e2e_request_latency_seconds", e2e_b, e2e_sum, e2e_count),
    ):
        lines.append(f"# TYPE {name} histogram")
        for b in _LAT_BUCKETS:
            lines.append(f'{name}_bucket{{{lbl},le="{b}"}} {buckets[b]}')
        lines.append(f'{name}_bucket{{{lbl},le="+Inf"}} {scount}')
        lines.append(f"{name}_sum{{{lbl}}} {ssum:.4f}")
        lines.append(f"{name}_count{{{lbl}}} {scount}")
    return "\n".join(lines) + "\n"


def main() -> None:
    print(f"serving-shim -> backend={OLLAMA_URL} model={MODEL} max_concurrency={MAX_CONCURRENCY} port={PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
