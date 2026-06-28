# Design — kube-ai

Distilled design reference. Read CLAUDE.md for safety invariants and gotchas.

---

## Component mapping: slurm-ai → kube-ai

| slurm-ai component | kube-ai component | Key difference |
|--------------------|-------------------|----------------|
| `slurm_exec.py` `SlurmCommandRunner` | `kubectl_exec.py` `KubectlCommandRunner` | kubectl instead of Slurm CLI |
| `collectors/slurm.py` `SlurmCollector` | `collectors/k8s.py` `K8sCollector` | scrapes vLLM /metrics + kubectl get deployment |
| `policy/engine.py` `PolicyEngine` | `policy/engine.py` `PolicyEngine` | composite formula instead of max(cpu,mem,io) |
| `tuner/aimd.py` `AimdTuner.next_max_jobs` | `tuner/aimd.py` `AimdTuner.next_replicas` + `next_max_num_seqs` | two tunables; direction inverted |
| `actuator/slurm.py` `SlurmActuator` | `actuator/k8s.py` `K8sActuator` | two paths; separate cooldowns; `kubectl scale` + `kubectl patch` |
| `config.py` (env only) | `config.py` (YAML + env override) | PyYAML; reads `config.yaml` then env on top |
| `types.py` `ClusterSnapshot` | `types.py` `ServingSnapshot` | vLLM-specific fields; sub-scores embedded |
| `metrics.py` `adaptive_*` | `metrics.py` `kube_ai_*` | same pattern, different prefix/names |

---

## ServingSnapshot fields

```python
@dataclass(slots=True)
class ServingSnapshot:
    timestamp: datetime
    desired_replicas: int
    ready_replicas: int
    available_replicas: int
    requests_waiting: int        # vllm:num_requests_waiting
    requests_running: int        # vllm:num_requests_running
    requests_swapped: int        # vllm:num_requests_swapped
    kv_cache_usage_perc: float   # vllm:kv_cache_usage_perc  [0..1]
    p95_ttft_sec: float          # vllm:time_to_first_token_seconds p95
    p50_ttft_sec: float          # vllm:time_to_first_token_seconds p50
    queue_pressure: float        # sub-score [0..1]
    cache_pressure: float        # sub-score [0..1]
    latency_pressure: float      # sub-score [0..1]
    current_max_num_seqs: int    # parsed from deployment env or config annotation
    metrics_available: bool      # False → saturation forced to 0.0
```

---

## Saturation formula

```
queue_pressure   = waiting / max(1, waiting + running)
cache_pressure   = vllm:kv_cache_usage_perc              (already [0..1])
latency_pressure = clamp((p95_ttft - TTFT_SLO) / TTFT_SLO, 0, 1)

saturation = 0.50 * queue_pressure
           + 0.30 * cache_pressure
           + 0.20 * latency_pressure

Hard override: if vllm:num_requests_swapped > 0:
    saturation = max(saturation, pressure_high + 0.01)
```

If `metrics_available` is False, saturation returns 0.0 (no action).

---

## Two-tunable actuator design

`K8sActuator.apply(decision: PolicyDecision)` gates two independent paths by `tune_mode`:

### Path A — Replicas (`tune_mode in {replicas, both}`)

- Cooldown: `cooldown_sec` (default 60 s)
- Command (live): `kubectl scale deployment <dep> -n <ns> --replicas=<N>`
- Bounds: `[min_replicas, max_replicas]`, enforced in tuner AND actuator
- AIMD: scale-out `current + 1`, scale-in `current // 2`

### Path B — Params (`tune_mode in {params, both}`)

- Cooldown: `param_cooldown_sec` (default 300 s, longer: param changes need vLLM restart)
- Command (live): `kubectl patch deployment <dep> -n <ns> --type=strategic -p '<json-patch>'`
  Patch sets `--max-num-seqs=<N>` in the container args.
- Bounds: `[min_max_num_seqs, max_max_num_seqs]`, enforced in tuner AND actuator
- AIMD: scale-out `current + 128`, scale-in `max(min, current // 2)`

### Dry-run behaviour (both paths)

Commands are built and logged with a `DRY_RUN` prefix. State is still advanced so the loop
simulates realistic behaviour without touching the cluster.

### State-on-success rule

State (`current_replicas`, `current_max_num_seqs`, `last_replica_apply`, `last_param_apply`)
is only updated when the kubectl command returns `ok=True`. A failed live command leaves state
unchanged; `last_apply` is not advanced; `changed=False`.

---

## kind sandbox + mock-vLLM strategy (commit 2+)

For integration testing without a real GPU cluster:
- Spin up a `kind` cluster with `kind create cluster --name vllm-test`
- Deploy a mock vLLM HTTP server (simple Flask/FastAPI app exposing `/metrics` in Prometheus text format)
- Point `vllm_metrics_url` at the mock; set `exec_mode: local` and `kubectl context: kind-vllm-test`
- The full loop runs end-to-end; only the GPU inference is fake

For commit 1, `vllm_mode: mock` uses a module-level static fixture that exercises the same parsers.
