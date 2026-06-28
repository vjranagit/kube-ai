# AGENTS.md — kube-ai

Sub-task map for parallel agent work. Each section is independently assignable.

---

## Single monkeypatch-target contract

All kubectl calls go through **`KubectlCommandRunner.run`** in `controller/kubectl_exec.py`.
Tests monkeypatch only this method — never `subprocess` directly, never individual kubectl helpers.

```python
# Pattern used in every actuator/collector test:
runner.run = lambda cmd, check=True: (True, FIXTURE_OUTPUT)
```

---

## File ownership map

| Component | File(s) | Owner notes |
|-----------|---------|-------------|
| Config | `controller/config.py`, `config.example.yaml`, `.env.example` | YAML+env read at import time |
| Types | `controller/types.py` | Pure dataclasses; no imports from other controller modules |
| Exec runner | `controller/kubectl_exec.py` | Only subprocess user; all modes |
| Metrics | `controller/metrics.py` | Prometheus Gauge singletons; `kube_ai_` prefix |
| Collector | `controller/collectors/k8s.py` | Parsers separated from source; mock vs real |
| Policy | `controller/policy/engine.py` | Pure/stateless; composite saturation formula |
| AIMD tuner | `controller/tuner/aimd.py` | Separate methods: `next_replicas`, `next_max_num_seqs` |
| Actuator | `controller/actuator/k8s.py` | Cooldown, bounds, dry-run, state-on-success |
| Loop | `controller/main.py` | Arg parse, build stack, loop, metrics |
| API | `apps/api/main.py` | FastAPI; `K8sCollector` only; no policy/tuner/actuator |
| Dashboard | `apps/dashboard/index.html` | Static; fetches `/serving/snapshot` |

---

## Parallelizable sub-tasks (commit 2+)

### A — Tests: collector parser unit tests
- Inputs: `_parse_vllm_metrics`, `_parse_deployment_json` functions
- Scope: edge cases (empty body, missing metrics, malformed JSON)
- Does NOT need a real cluster

### B — Tests: policy engine
- Inputs: `PolicyEngine.saturation_score`, swap override
- Scope: composite formula, boundary values, swapped override

### C — Tests: AIMD tuner
- Inputs: `AimdTuner.next_replicas`, `AimdTuner.next_max_num_seqs`
- Scope: scale-out/in/hold, bounds clamp, boundary saturation values

### D — Tests: actuator dry-run / live / cooldown
- Inputs: `K8sActuator.apply`
- Scope: dry-run logs, cooldown skip, per-path cooldown, state-on-success, changed flag
- Monkeypatch target: `KubectlCommandRunner.run`

### E — Tests: API endpoints
- Inputs: `apps.api.main:app`
- Scope: `/healthz`, `/serving/snapshot`, `/deployment/status`
- Uses `fastapi.testclient.TestClient`

### F — RL tuner (commit 3)
- New file: `controller/tuner/rl.py`
- `build_tuner` already returns `AimdTuner` for `tuner_kind=rl` with a TODO; replace that stub

---

## Test-first rule

Write tests before any refactor. The monkeypatch contract means every component can be unit-tested
without a cluster. New features must ship with tests; failing tests block merge.
