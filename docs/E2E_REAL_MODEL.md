# E2E — real-model validation (not mock)

kube-ai's control loop, driven by a **real LLM under real concurrent load**, makes correct scaling
decisions and issues real `kubectl` actions. This complements the mock-vLLM e2e (`E2E_RESULTS.md`)
and the live kind scale-out/in run (`E2E_REAL_VLLM.md` covers the vLLM-CPU path on supported HW).

## Why a serving shim instead of vLLM directly

vLLM requires GPU compute capability ≥ 7.0 **or** an AVX2/AVX-512 CPU. The hardware available for
this run had neither: the reachable GPU is a Tesla P4 (Pascal, SM 6.1) and the CPUs are pre-AVX2
(the prebuilt vLLM-CPU image aborts with `SIGILL`). A capable GPU/host (e.g. an RTX-class node) was
offline. So the real model was served by **Ollama** (llama.cpp — runs on Pascal and old CPUs), and a
thin **serving-shim** (`infra/serving-shim/app.py`) sits in front of it: it forwards real
`/v1/completions` traffic to the model and measures genuine serving pressure — in-flight vs queued
requests against a capacity limit, plus real latency — exporting it as native `vllm:*` metrics.
kube-ai's collector scrapes it **unmodified**. The vLLM-CPU manifest
(`infra/k8s/real-vllm-cpu-deployment.yaml`) is kept for AVX2/AVX-512 hosts.

## Setup

- **Model**: `qwen2.5:0.5b` served by Ollama, GPU-accelerated on an **NVIDIA Tesla P4** (`<gpu-host>`).
- **Shim**: `infra/serving-shim/app.py`, `MAX_CONCURRENCY=4`, exposing `vllm:*` metrics.
- **Load**: `scripts/load-real.py` — concurrent `/v1/completions` at tiers **2 → 8 → 16**, real generations.
- **Controller**: `vllm_mode=real`, `vllm_metrics_url` → shim, `dry_run=false`, `tune_mode=both`,
  `cooldown_sec=5`, `max_replicas=3`. Replica/param actuation runs against the kind `vllm-server`
  Deployment (real `kubectl scale`/`patch`). The serving + pressure are real; the scaled Deployment
  is the in-cluster stand-in, since the model runs on the GPU host outside the cluster.

## Result — saturation tracks the real model, controller reacts correctly

| phase | requests_waiting | requests_running | kv_cache | saturation | decision | target replicas |
|------:|-----------------:|-----------------:|---------:|-----------:|----------|----------------:|
| idle/low | 0 | 2 | 0.50 | 0.15 | scale_in | → 1 |
| ramp | 4 | 4 | 0.50 | 0.55 → 0.73 | hold | 1 |
| heavy | 4 | 4 | 1.00 | 0.75 | scale_out | 1 → 2 → 3 |
| load drop | 0 | 2 | — | 0.35 | scale_in | → 1 |
| heavy again | 12 | 4 | 1.00 | 0.875 | scale_out | 1 → 2 → 3 |

- `metrics_available=true` for every tick — the collector ingested real metrics from the live model.
- Final cluster state: `kubectl get deploy vllm-server` → `spec=3 ready=2` (scaled out under load).
- Saturation rose monotonically with real concurrency and fell when load stopped; the AIMD tuner
  scaled out (+1, to the `max_replicas=3` ceiling) under pressure and scaled in (÷2) when idle —
  exactly as designed, **driven by a real model rather than a fixture**.

## Verdict

| Check | Result |
|-------|--------|
| Real model serving real generations | PASS (`qwen2.5:0.5b` on Tesla P4) |
| Collector ingests real metrics under real load | PASS (`metrics_available=true`) |
| Saturation tracks real concurrency | PASS (0.15 idle → 0.875 heavy) |
| Scale-out under real pressure | PASS (1 → 3, real `kubectl scale`) |
| Scale-in when idle | PASS (→ 1) |
| Bounds respected | PASS (never exceeded `max_replicas=3`) |

## Reproduce

```bash
# 1. serve a real model (any OpenAI-compatible backend; Ollama shown):
#    on the GPU host:  docker run -d --gpus all -p 11500:11434 ollama/ollama
#                      docker exec <c> ollama pull qwen2.5:0.5b
# 2. shim -> backend:
OLLAMA_URL=http://<gpu-host>:11500 MODEL=qwen2.5:0.5b MAX_CONCURRENCY=4 PORT=18000 \
    python3 infra/serving-shim/app.py
# 3. drive real batch/stress load:
python3 scripts/load-real.py --base-url http://localhost:18000 --model qwen2.5:0.5b --tiers 2,8,16
# 4. point kube-ai at the shim (vllm_mode=real, vllm_metrics_url=http://localhost:18000/metrics)
#    and run the controller against your cluster.
```
