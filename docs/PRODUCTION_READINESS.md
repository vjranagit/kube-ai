# Production Readiness — kube-ai v0.1.0

Date: 2026-06-28

---

## VERIFIED

### Test suite

| Check | Result |
|-------|--------|
| Unit tests (with config.yaml absent — clean checkout) | **331 passed, 0 failed** |
| Unit tests (with live config.yaml present — e2e mode) | **331 passed, 0 failed** (test isolation via conftest monkeypatching) |
| ruff check (lint + format) | **0 violations** |

### Live kind end-to-end (cluster: kind-kube-ai, 2026-06-28)

See `docs/E2E_RESULTS.md` for the full benchmark table and tick-by-tick logs.

| Verdict | Check | Evidence |
|---------|-------|---------|
| PASS | Live scrape from `vllm_metrics_url` | `curl localhost:30080/metrics` returns real `vllm:*` metrics; controller tick snapshots show live values |
| PASS | Scale-out: 1→2→3→4→5 replicas under high load (saturation≈0.90) | AIMD additive increase; each `kubectl scale` returned OK |
| PASS | Scale-in: 4→2→1 replicas at zero load (saturation=0.0) | AIMD multiplicative decrease (`÷2`); each `kubectl scale` returned OK |
| PASS | `--max-num-seqs` tuned 128→256→384→512→640→768 | `kubectl patch` args-preserving; logged in `tmp/controller_high.log` |
| PASS | Bounds never violated | Replicas always in [1, 5]; max_num_seqs always in [128, 1024]; double-clamped |
| PASS | State advances only on success | Actuator state updated only when `runner.run()` returns `ok=True`; confirmed in unit tests and live logs |

Benchmark numbers (scale-out run, load=0.95):
- Tick 1: saturation=0.900, waiting=76, replicas 1→2
- Tick 5: saturation=0.900, waiting=76, replicas 4→5
- Tick 8: saturation=0.881, waiting=69, replicas=5 (at max, hold)

### Security / audit

All Critical (C1–C5) and High (H1–H6) findings from the security audit resolved.
See `docs/WORK_LOG.md` (2026-06-28 Audit remediation section) for the full fix log.

Medium and Low quick-fixes (M1, M2, L1, L3) also applied.

### UI smoke test (2026-06-28, uvicorn port 5500 — port 8080 occupied by Docker on this host)

| Endpoint | Result | Notes |
|----------|--------|-------|
| `GET /healthz` | PASS | `{"status": "ok"}` |
| `GET /ui/` | PASS | HTTP 200; HTML with Chart.js dashboard content |
| `GET /api/state` | PASS | JSON with `metrics_available=true`, `replicas=2` (live kind cluster), full gauges object |
| `GET /api/config` | PASS | Full config JSON with 14 whitelisted fields |
| Screenshot | SKIPPED | playwright not installed; not installing heavy deps |

---

## KNOWN LIMITATIONS / DEFERRED

| Item | Severity | Status |
|------|----------|--------|
| Real vLLM on GPU hosts (a dedicated GPU host / a CPU-heavy host with small GPU) not exercised | Medium | Deferred — mock-vLLM used for all e2e; go-live checklist below |
| RBAC on config UI endpoints | Medium | Deferred — `# RBAC TODO` hooks in place at every new endpoint |
| M3: `_gauge_val` uses private prometheus `_value.get()` API | Medium | Deferred — requires prometheus_client refactor when version pinned |
| M4: E2E_RESULTS.md clarification (the doc now contains full verdicts) | Low | Resolved by this release |
| M5: `_percentile_from_buckets` returns 0.0 fallback instead of None | Medium | Deferred — requires Optional[float] propagation throughout snapshot consumers |
| L2: Controller Deployment manifest missing resource limits | Low | Deferred — operational concern; add to `infra/k8s/controller-deployment.yaml` before prod |

Reference: `docs/MORNING_REVIEW.md` for full deferred-item rationale.

---

## GO-LIVE CHECKLIST — pointing at real vLLM on <gpu-host>

Follow these steps to switch from mock-vLLM to a real vLLM GPU host.

### 1. Build and start vLLM on the GPU host

```bash
# On <gpu-host> (GPU) — adjust model as needed
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model mistralai/Mistral-7B-v0.1 \
  --served-model-name mistral
```

Verify the Prometheus endpoint is live:
```bash
curl http://<gpu-host>:8000/metrics | grep vllm:num_requests_waiting
```

### 2. Probe connectivity from the controller host

```bash
./scripts/check-remote.sh  # verifies vllm_metrics_url reachability and metric format
```

### 3. Configure the controller

```yaml
# config.yaml
vllm_mode: real
vllm_metrics_url: http://<gpu-host>:8000/metrics
exec_mode: ssh
ssh_host: <gpu-host>
ssh_user: <user>
ssh_key_file: ~/.ssh/id_ed25519
vllm_deployment: vllm-server
vllm_namespace: default
context: <your-prod-kube-context>
dry_run: true          # ALWAYS start in dry-run
tune_mode: replicas    # start with replicas only; add params after baseline
```

### 4. Run in dry-run first

```bash
python -m controller.main --dry-run true --interval 10 --max-iterations 20
# Watch logs: confirm saturation scores look realistic, no errors
```

### 5. Monitor in Grafana

- Start Prometheus + Grafana: `docker compose -f infra/docker/docker-compose.yml up -d`
- Open http://localhost:3000 (admin/admin)
- Confirm `kube_ai_saturation_score`, `kube_ai_requests_waiting`, `kube_ai_kv_cache_usage_perc`
  are live and non-zero under load.

### 6. Enable live mutations

```yaml
dry_run: false
```

Restart controller. Watch Grafana for replica changes and max_num_seqs progression.

### 7. Enable param tuning (optional)

```yaml
tune_mode: both
param_cooldown_sec: 600   # generous cooldown; vLLM restarts on patch
```

Monitor for unexpected restarts. Tune `param_cooldown_sec` to taste.
