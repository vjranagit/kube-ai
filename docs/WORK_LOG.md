# kube-ai work log

## 2026-06-28 — Commit 2: test suite

Added a comprehensive pytest suite under `controller/tests/` mirroring the slurm-ai discipline (single kubectl seam, subprocess config tests, stress invariants).

### Files added
| File | Tests | Description |
|------|-------|-------------|
| `controller/tests/__init__.py` | — | Package marker |
| `controller/tests/conftest.py` | — | Shared fixtures: `make_cfg`, `FakeRunner`, `patch_runner`, `MockMetricsServer`, canned metric constants |
| `controller/tests/test_config.py` | 39 | Defaults and env overrides via subprocess; TUNE_MODE values; bounds sanity |
| `controller/tests/test_kubectl_exec.py` | 24 | Command building for local/ssh/docker modes; (ok, out) contract; timeout handling |
| `controller/tests/test_collector.py` | 40 | `_parse_vllm_metrics` (gauges + histogram p95/p50), `_parse_deployment_json`, `snapshot()` with monkeypatched runner + MockMetricsServer; edge cases: empty metrics, unreachable endpoint, malformed JSON |
| `controller/tests/test_policy_engine.py` | 26 | Saturation weighted formula exactness; swap override; `metrics_available=False`; sub-score clamping |
| `controller/tests/test_aimd.py` | 21 | `next_replicas` + `next_max_num_seqs`: scale-out/scale-in/hold branches; boundary values |
| `controller/tests/test_aimd_full.py` | 9 | 1 000-iteration invariants: bounds, int type, convergence, no exceptions |
| `controller/tests/test_actuator.py` | 32 | dry-run (no runner calls), live mode, TUNE_MODE gating, per-path cooldowns, double-clamp, changed flag, **state advances ONLY on ok=True** (key safety regression) |
| `controller/tests/test_stress.py` | 5 | 2 000-iteration end-to-end loop (policy → tuner → actuator); bounds and saturation invariants under random pressure |
| `controller/tests/test_api.py` | 28 | FastAPI TestClient: `/`, `/healthz`, `/serving/snapshot`, `/deployment/status`, `/metrics` (kube_ai_* text), `/ui/` (HTML) |

### Total: 234 tests passing, 0 failing

### Bugs found and fixed
None found in controller code. All modules matched their documented contracts exactly.
`ruff check .` passes with no issues.

## 2026-06-28 — Build wave (sandbox, UI, RL)
- feat(sandbox) b1a409e: kind + mock-vLLM (real vllm:* metrics) + k8s manifests + Prometheus/Grafana + scripts. deployment/container=vllm-server, ns=kube-ai, NodePort 30080.
- feat(ui) 1dfd9e1: Chart.js live dashboard + config editor + loop start/stop; 65 api+ui tests green.
- feat(tuner) 36f22bb: tabular Q-learning RLTuner (replicas + max_num_seqs) + ServingSimulator + training; qtable tracked. Full suite 286 passing.
- Next: kind e2e + stress + benchmark, then audit + gap passes.

## 2026-06-28 — e2e milestone (commit 87dc95e)

6/6 live verdicts PASS on the kind sandbox:
- Scale-in 4→2→1 replicas under low pressure; scale-out 1→5 replicas under high pressure.
- max_num_seqs tuned 128→768 in response to KV-cache saturation.
- Bounds respected (min_replicas=1, max_replicas=5, min_max_num_seqs=128, max_max_num_seqs=1024).
- 294 tests green in a clean checkout (no config.yaml present).

## 2026-06-28 — Gap fixes: test isolation + loop hardening

### Test isolation (12 failures fixed)

Root cause: `controller/config.py` loads `_YAML` once at module import time from `config.yaml`
(or `KUBE_AI_CONFIG`). When a live `config.yaml` is present (e2e / sandbox mode):
- `apps/api/main.py` calls `ControllerConfig()` at import time → wrong defaults → 3 test_api/ui failures.
- `test_config.py` subprocesses inherit CWD=repo root, strip `KUBE_AI_CONFIG`, then find
  the live `config.yaml` → 9 subprocess test failures.

Fix applied:
- `controller/tests/conftest.py`: reset `controller.config._YAML = {}` at conftest import time
  (before any test module imports `apps.api.main`); plus autouse per-test fixture that
  monkeypatches `_YAML` to `{}` for every test function.
- `controller/tests/test_config.py`: `run_cfg_expr()` sets
  `KUBE_AI_CONFIG=/nonexistent/kube-ai-test-config.yaml` in the subprocess env (after
  stripping other config keys) so the subprocess never loads a live yaml file.

Verified: `pytest -q` 297 passed, 0 failed — both with and without `config.yaml` present.

### Loop hardening (minimal, well-tested changes)

- `controller/kubectl_exec.py`: added `except Exception` fallthrough so `run()` truly
  never raises — catches `ValueError` from `_build()` (e.g. SSH/docker mode misconfiguration).
  2 new tests in `test_kubectl_exec.py`.
- `controller/main.py`: wrapped each tick body in `try/except Exception` (logs and continues);
  added SIGTERM/SIGINT handler via `threading.Event` so `_shutdown.wait()` replaces
  `time.sleep()` and exits immediately on signal. `--max-iterations` behaviour unchanged.
- `controller/actuator/k8s.py`: `_sync_initial_state()` already best-effort (try/except).
  Added 2 explicit tests in `test_actuator.py` (runner returns False; runner raises).
- `controller/collectors/k8s.py`: already returns `metrics_available=False` on any fetch
  failure (existing tests confirm). No code change needed.
