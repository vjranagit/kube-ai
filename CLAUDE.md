# CLAUDE.md — kube-ai

This file provides guidance to Claude Code when working with code in this repository.

## What this is

`kube-ai` — a Kubernetes adaptive control loop that auto-tunes vLLM serving deployments.
Each interval it reads vLLM queue/cache/latency signals, computes a saturation score, and
(outside dry-run) scales replicas and/or adjusts `--max-num-seqs` to keep the serving stack
within SLO. Inspired by slurm-ai and the SelfTune philosophy (NSDI 2023). Python 3.11+.

---

## ⚠️ DIRECTION INVERSION — READ THIS FIRST ⚠️

This project is the **opposite** of slurm-ai w.r.t. what "high saturation" means:

| Project   | High saturation → action | Low saturation → action |
|-----------|--------------------------|-------------------------|
| slurm-ai  | decrease max_jobs (throttle) | increase max_jobs (allow more) |
| **kube-ai** | **SCALE OUT** (add replicas / raise max_num_seqs) | **SCALE IN** (shed replicas / lower max_num_seqs) |

AIMD asymmetry is inverted accordingly:
- Scale-out step: `+1` replica (additive-increase, conservative)
- Scale-in step: `current // 2` replicas (multiplicative-decrease, aggressive)

Do NOT confuse these directions. Every reviewer should check this.

---

## 6-step control loop

```
collect → saturation_score → tuner → actuate → set metrics → sleep
```

1. **collect** — `K8sCollector.snapshot()` scrapes vLLM `/metrics` (Prometheus text) and runs
   `kubectl get deployment -o json` through `KubectlCommandRunner`.
2. **saturation_score** — `PolicyEngine.saturation_score(snap)` computes a composite [0,1]:
   `0.50*queue_pressure + 0.30*cache_pressure + 0.20*latency_pressure`.
   Hard override: if `vllm:num_requests_swapped > 0`, floor saturation just above PRESSURE_HIGH.
   If metrics unavailable → return 0.0 (no action into a broken cluster).
3. **tuner** — `AimdTuner` computes target replicas and target `max_num_seqs` independently.
4. **actuate** — `K8sActuator.apply(decision)` gates by `tune_mode`, double-clamps bounds,
   enforces per-path cooldowns, and (outside dry-run) runs kubectl commands.
5. **set metrics** — Prometheus `Gauge` singletons updated each tick.
6. **sleep** — wait `interval_sec` before the next iteration.

---

## Single subprocess choke point

**ALL kubectl calls go through `controller/kubectl_exec.py` `KubectlCommandRunner.run(cmd, check)`.**

Signature: `run(cmd: str, check: bool = True) -> tuple[bool, str]`

- Returns `(ok: bool, output: str)`. Never raises on command failure.
- Modes: `local` (subprocess on host), `ssh` (SSH tunnel to remote), `docker` (docker exec).
- Any new kubectl interaction MUST go through this class. Never call `subprocess` directly elsewhere.

---

## TUNE_MODE

Controlled by `tune_mode` config field (env: `TUNE_MODE`). Accepted values:

| Value      | Effect |
|------------|--------|
| `replicas` | Only scale replicas; `max_num_seqs` unchanged |
| `params`   | Only adjust `max_num_seqs` via `kubectl patch`; replicas unchanged |
| `both`     | Apply both (replicas first, params second) |

Replica changes use `cooldown_sec`. Param changes use `param_cooldown_sec` (longer, default 300s).

---

## VLLM_MODE

`vllm_mode: mock | real` (env: `VLLM_MODE`).

- `mock` (default) — collector returns a static fixture; no real cluster or vLLM needed.
  The parsers (`_parse_vllm_metrics`, `_parse_deployment_json`) run on the fixture exactly as
  they will on real data, so later commits only swap the source, not the parser.
- `real` — collector scrapes `vllm_metrics_url` via `urllib.request` and runs kubectl.

---

## Safety invariants (do not weaken)

- `dry_run` defaults to `true`; only explicit `--dry-run false` / `CONTROLLER_DRY_RUN=false` mutates k8s.
- Replica bounds `[min_replicas, max_replicas]` enforced in BOTH tuner AND actuator.
- `max_num_seqs` bounds `[min_max_num_seqs, max_max_num_seqs]` enforced in BOTH tuner AND actuator.
- Cooldown lives in `K8sActuator.apply()`. Replica path uses `cooldown_sec`; param path uses `param_cooldown_sec`.
- State advances ONLY on success. Failed live commands do not update `last_apply` or state.
- No destructive ops: no `kubectl delete`, no `kubectl drain`. Only `scale` and `patch`.
- If metrics unavailable → `metrics_available=False` → saturation 0.0 → no action.

---

## Commands

```bash
# Install
pip install -e ".[dev]"

# Lint
ruff check .

# Tests (commit 2+)
pytest -q

# Control loop (offline, dry-run, 3 ticks)
python -m controller.main --dry-run true --interval 1 --max-iterations 3

# API / dashboard (separate process)
uvicorn apps.api.main:app --host 127.0.0.1 --port 8080

# Metrics endpoint (loop process)
# Exposed on :9108 via prometheus_client.start_http_server
```

---

## Config

Config is loaded from a YAML file (default `config.yaml` at repo root) then ENV overrides are
applied on top, all at **import time**. Set env vars before importing the package; there is no
`load_config()` function.

See `config.example.yaml` for all fields with defaults and comments.
See `.env.example` for the env override names.

---

## Gotchas

- `ControllerConfig` reads the YAML file AND env at import time. Set env before importing.
- `vllm_mode=mock` is the default; set `vllm_mode: real` in config.yaml (or `VLLM_MODE=real`) for live.
- Prometheus metric name: `vllm:kv_cache_usage_perc` (NOT `vllm:gpu_cache_usage_perc` — that name changed).
- `tune_mode=both` applies replicas first (outer cooldown), then params (inner, longer cooldown).
- AIMD for replicas: scale-out is `+1`, scale-in is `//2` (opposite of slurm-ai).
