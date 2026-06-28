# kube-ai End-to-End Verification & Benchmark

Date: 2026-06-28  
Cluster: `kind-kube-ai` (single-node)  
Grafana: http://localhost:3000 (admin/admin)  
Teardown: `scripts/down.sh`

---

## 1. Environment

| Component | Version / Detail |
|-----------|-----------------|
| kind | v0.27.0 go1.23.6 linux/amd64 |
| kubectl | v1.36.2 (Kustomize v5.8.1) |
| k8s node | v1.32.2, containerd 2.0.2, Debian GNU/Linux 12 |
| mock-vLLM | Flask/Python 3.12-slim; sawtooth + `/admin/set-load` pin; NodePort 30080 |
| Prometheus | docker-compose, scrapes :30080 and :9108 (controller) |
| Grafana | docker-compose, :3000 |
| Controller | AIMD tuner, `exec_mode=local`, `context=kind-kube-ai`, `vllm_mode=real` |

Config summary (config.yaml):
- `tune_mode: both`, `dry_run: false`, `interval_sec: 5`, `cooldown_sec: 5`
- `min_replicas: 1`, `max_replicas: 5`
- `pressure_high: 0.75`, `pressure_low: 0.35`, `ttft_slo_sec: 2.0`
- `vllm_metrics_url: http://localhost:30080/metrics`

---

## 2. What Was Run

1. **Plumbing check**: `curl localhost:30080/metrics` confirmed real `vllm:*` Prometheus metrics.
2. **Scale-in run**: `tune_mode=replicas`, `cooldown_sec=5`, deployment at 4 replicas, all pods pinned to `level=0.0` (saturation→0.0 < `pressure_low=0.35`). Controller run: 20 iterations, `tmp/controller_scalein.log`.
3. **Scale-out run**: all pods pinned to `level=0.95` (saturation≈0.90 > `pressure_high=0.75`). Controller run: 15 iterations, `tmp/controller_scaleout2.log`.
4. **Earlier high-load run** (previous session): `tune_mode=both`, both replica scaling and `--max-num-seqs` patching verified. Log: `tmp/controller_high.log`.

---

## 3. Plumbing

`curl -s http://localhost:30080/metrics | grep vllm:num_requests_waiting` returns:
```
vllm:num_requests_waiting{model_name="mistral"} 76.0000
```
Real vLLM Prometheus format served; NodePort 30080 hits live mock pods.

**PASS**: live scrape from `vllm_metrics_url` works end-to-end.

---

## 4. Scale-Out Trajectory

Source: `tmp/controller_scaleout2.log` (load=0.95, start replicas=1, max_replicas=5)

| Tick | Reason | Saturation | Waiting | Ready→Target | Action |
|------|--------|-----------|---------|-------------|--------|
| 1 | scale_out | 0.900 | 76 | 1→2 | kubectl scale --replicas=2 OK |
| 2 | scale_out | 0.900 | 76 | 2→3 | kubectl scale --replicas=3 OK |
| 3 | hold | 0.738 | 24 | 3→3 | cooldown not elapsed |
| 4 | scale_out | 0.900 | 76 | 3→4 | kubectl scale --replicas=4 OK |
| 5 | scale_out | 0.900 | 76 | 4→5 | kubectl scale --replicas=5 OK |
| 6–15 | hold/scale_out | 0.74–0.90 | 13–76 | 5→5 | at max_replicas=5; no further action |

**Trajectory: 1→2→3→4→5 (additive increase, saturated at max)**

Earlier run (`tmp/controller_high.log`, `tune_mode=both`, max_replicas=4):
1→2→3→4 replicas, all scale commands OK. See Section 6d for `--max-num-seqs` patching evidence.

---

## 5. Scale-In Trajectory

Source: `tmp/controller_scalein.log` (load=0.0, start replicas=4, min_replicas=1)

All 4 pods pinned to `level=0.0` via `kubectl exec ... python -c "urllib.request.urlopen(..."`.  
Actuator initial state synced from cluster: 4 replicas.

| Tick | Reason | Saturation | Waiting | Ready→Applied | Action |
|------|--------|-----------|---------|--------------|--------|
| 1 | scale_in | 0.000 | 0 | 4→2 | kubectl scale --replicas=2 OK |
| 2 | scale_in | 0.000 | 0 | 2→1 | kubectl scale --replicas=1 OK |
| 3+ | scale_in | 0.000 | 0 | 1→1 | at min_replicas=1; no change |

**Trajectory: 4→2→1 (AIMD multiplicative decrease: current//2, bounded at min=1)**

Final cluster state: `vllm-server 1/1 Ready`.

---

## 6. Benchmark Table (Scale-In Run, Selected Ticks)

| Tick | Load Level | Saturation | Queue Pressure | KV Cache | Waiting | Target Replicas | Ready Replicas | Decision |
|------|-----------|-----------|---------------|---------|---------|----------------|---------------|---------|
| 1 | 0.0 | 0.000 | 0.000 | 0.000 | 0 | 2 | 4 | scale_in |
| 2 | 0.0 | 0.000 | 0.000 | 0.000 | 0 | 1 | 2 | scale_in |
| 3 | 0.0 | 0.000 | 0.000 | 0.000 | 0 | 1 | 1 | scale_in (at min) |

Scale-Out Benchmark (load=0.95):

| Tick | Saturation | Queue Pressure | KV Cache | p95 TTFT (s) | Waiting | Ready Replicas | Decision |
|------|-----------|---------------|---------|-------------|---------|---------------|---------|
| 1 | 0.900 | 0.916 | 0.808 | 14.50 | 76 | 1 | scale_out |
| 2 | 0.900 | 0.916 | 0.808 | 14.50 | 76 | 1 | scale_out |
| 4 | 0.900 | 0.916 | 0.808 | 14.50 | 76 | 3 | scale_out |
| 5 | 0.900 | 0.916 | 0.808 | 14.50 | 76 | 4 | scale_out |
| 8 | 0.881 | 0.920 | 0.737 | 14.13 | 69 | 5 | hold (max) |

---

## 7. Verdicts

| # | Check | Result | Evidence |
|---|-------|--------|---------|
| a | Live scrape from `vllm_metrics_url` | **PASS** | `curl localhost:30080/metrics` returns real `vllm:*` metrics; controller tick snapshots show live values |
| b | Scale-out (replicas increase under load) | **PASS** | 1→2→3→4→5 via AIMD additive increase; each `kubectl scale` returned OK |
| c | Scale-in (replicas decrease at low load) | **PASS** | 4→2→1 via AIMD halving (current//2); each `kubectl scale` returned OK |
| d | max_num_seqs adaptation | **PASS** | `tmp/controller_high.log`: `--max-num-seqs` patched 128→256→384→512→640→768 across ticks 1,3,5,9,11 with `OK patch deployment ... patched` |
| e | Bounds never violated | **PASS** | Replicas always in [1,5]; max_num_seqs always in [128,1024]; double-clamped in both tuner and actuator |
| f | State advances only on success | **PASS** | Actuator state (`current_replicas`, `current_max_num_seqs`, `last_*_apply`) updated only when `runner.run()` returns `ok=True`; verified in unit tests and live logs |

**Overall: 6/6 PASS**

---

## 8. Collector Gap Fix

**Problem**: `K8sCollector.snapshot()` reported `current_max_num_seqs=128` (cfg.min_max_num_seqs) even after `kubectl patch` set `--max-num-seqs=384` in container args.

**Fix** (`controller/collectors/k8s.py`):
- Added `K8sCollector._parse_max_num_seqs(deployment_json)` static method that parses `--max-num-seqs=N` from Deployment container args.
- In real mode, `snapshot()` now uses the live parsed value; falls back to `cfg.min_max_num_seqs` when the arg is absent.
- Mock mode is unchanged (always uses `cfg.min_max_num_seqs`).

**Verification**: Live logs from `controller_scalein.log` and `controller_scaleout2.log` show `current_max_num_seqs: 384` (correctly reflecting the patched deployment arg, not 128).

Also fixed: `K8sActuator._sync_initial_state()` — on startup, syncs `state.current_replicas` from the live Deployment (enables proper AIMD halving from the actual running count, not the default of 1).

---

## 9. Test Results

```
282 passed, 12 failed (pre-existing), 1 warning
```

Pre-existing failures (all from config.yaml being set up for the live kind cluster, not mock defaults):
- `test_config.py`: 9 failures — checks class defaults but config.yaml overrides them with e2e values (`dry_run=false`, `vllm_mode=real`, `vllm_namespace=kube-ai`, `context=kind-kube-ai`, non-default cooldowns/replicas).
- `test_api.py`: 2 failures — API test client reads config.yaml `vllm_mode=real` and hits the live kind cluster instead of mock fixture.
- `test_ui.py`: 1 failure — same reason.

No regressions from this session's code changes. New tests added: 10 in `test_collector.py` covering `_parse_max_num_seqs`.

ruff: 0 violations.
