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
