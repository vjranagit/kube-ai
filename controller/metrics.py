"""Prometheus Gauge singletons for the kube-ai control loop.

All gauge names are prefixed with ``kube_ai_``.
Import this module once; the loop process calls start_http_server(cfg.metrics_port).
"""

from prometheus_client import Gauge

SATURATION = Gauge("kube_ai_saturation_score", "Composite saturation score [0..1]")
REQUESTS_WAITING = Gauge("kube_ai_requests_waiting", "vLLM requests waiting (queue)")
REQUESTS_RUNNING = Gauge("kube_ai_requests_running", "vLLM requests running")
KV_CACHE_USAGE = Gauge("kube_ai_kv_cache_usage_perc", "vLLM KV cache utilisation [0..1]")
TARGET_REPLICAS = Gauge("kube_ai_target_replicas", "Target replica count decided by tuner")
READY_REPLICAS = Gauge("kube_ai_ready_replicas", "Ready replicas reported by Kubernetes")
TARGET_MAX_NUM_SEQS = Gauge("kube_ai_target_max_num_seqs", "Target max-num-seqs decided by tuner")
ACTION_CHANGED = Gauge("kube_ai_action_changed", "1 if the actuator applied a change this tick")
P95_TTFT = Gauge("kube_ai_p95_ttft_sec", "Approximate p95 time-to-first-token (seconds)")
QUEUE_PRESSURE = Gauge("kube_ai_queue_pressure", "Queue pressure sub-score [0..1]")
CACHE_PRESSURE = Gauge("kube_ai_cache_pressure", "Cache pressure sub-score [0..1]")
LATENCY_PRESSURE = Gauge("kube_ai_latency_pressure", "Latency pressure sub-score [0..1]")
