# Research Notes — kube-ai

## Attribution

This is an independent, open-source educational/portfolio implementation inspired by publicly
available papers. All code was written from scratch from public sources. It is **not** affiliated
with, derived from, or representative of any proprietary or internal system, and uses no
non-public information.

---

## Primary references

### SelfTune (NSDI 2023)
- Paper: *SelfTune: Tuning Cluster Managers* — USENIX NSDI 2023
  <https://www.usenix.org/conference/nsdi23/presentation/karthikeyan>
- Video: <https://www.youtube.com/watch?v=DfU2yx1XW8A>
- Core idea: measure, adjust, observe, repeat — a continuous feedback loop for cluster parameter
  tuning without requiring a model of the workload.
- Our use: the 6-step loop (collect → score → tune → actuate → metrics → sleep) and the
  AIMD-first philosophy directly follow this pattern.

### Chiron (arXiv 2501.08090)
- Paper: *Chiron: Accelerating Long-Context LLM Inference with Request Migration and Recomputation*
  arXiv 2501.08090
- Relevance: discusses KV-cache pressure and request preemption (swapped requests) as first-class
  signals in LLM serving. Informs our use of `vllm:num_requests_swapped` as a hard override
  signal: any non-zero swapped count means the cluster is already preempting requests, which is
  more severe than the composite score alone would indicate.

---

## vLLM metric names and semantics

Verified against vLLM source (`vllm/engine/metrics.py`) and Prometheus output from a running instance.

| Metric name | Type | Semantics |
|-------------|------|-----------|
| `vllm:num_requests_waiting` | Gauge | Requests queued but not yet scheduled |
| `vllm:num_requests_running` | Gauge | Requests currently being processed by the engine |
| `vllm:num_requests_swapped` | Gauge | Requests preempted and swapped to CPU/disk; non-zero = KV pressure |
| `vllm:kv_cache_usage_perc` | Gauge | KV cache utilization [0..1] — use this name, NOT `gpu_cache_usage_perc` (old name, removed) |
| `vllm:time_to_first_token_seconds` | Histogram | Time from request arrival to first generated token |
| `vllm:e2e_request_latency_seconds` | Histogram | End-to-end request latency |
| `vllm:request_success_total` | Counter | Successfully completed requests |
| `vllm:generation_tokens_total` | Counter | Total generated tokens |

**Important name change**: `vllm:gpu_cache_usage_perc` was renamed to `vllm:kv_cache_usage_perc`
in vLLM >= 0.4.x. Always use `vllm:kv_cache_usage_perc`.

### p95 TTFT extraction

`vllm:time_to_first_token_seconds` is a Prometheus histogram. p95 is approximated by finding
the smallest bucket `le` where the cumulative count ≥ 0.95 * total count. In the static fixture
and the parser, we read `_bucket` lines and compute this directly. If fewer than 2 buckets are
available, p95 falls back to `+Inf` boundary or 0.0.

---

## AIMD inversion rationale

In slurm-ai, high saturation → *reduce* `max_jobs` (throttle submission).
In kube-ai, high saturation → *increase* replicas (scale out capacity).

The control direction is inverted because:
- slurm-ai controls demand (how many jobs can run concurrently)
- kube-ai controls supply (how many serving replicas handle the load)

AIMD asymmetry is designed to be **conservative on scale-out** (`+1`) and
**aggressive on scale-in** (`// 2`). This matches cloud cost intuition: adding a GPU replica is
expensive and should be done carefully; releasing idle replicas quickly is beneficial.

For `max_num_seqs`:
- Scale-out: `current + 128` (larger step; max_num_seqs is just a parameter, not a new pod)
- Scale-in: `max(min_bound, current // 2)`

---

## Out of scope (commit 1)

- RL tuner (commit 3)
- kind sandbox + mock-vLLM integration tests (commit 2)
- Multi-cluster / multi-deployment support
- GPU memory-aware replica sizing
- Horizontal pod autoscaler (HPA) integration or replacement
