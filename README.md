# kube-ai

`kube-ai` is a Kubernetes adaptive control loop that auto-tunes vLLM serving deployments.
Each interval it reads queue pressure, KV-cache utilization, and TTFT latency from vLLM's
Prometheus endpoint, computes a composite saturation score, and — outside dry-run — scales
replicas via `kubectl scale` and/or adjusts `--max-num-seqs` via `kubectl patch`. All mutations
are bounded, cooled-down, and gated behind a dry-run default.

> **Attribution & scope.** Independent educational/portfolio implementation inspired by
> *SelfTune: Tuning Cluster Managers* (USENIX NSDI 2023,
> <https://www.usenix.org/conference/nsdi23/presentation/karthikeyan>) and the Chiron paper
> (arXiv 2501.08090). Written from scratch using public papers. Not affiliated with any
> proprietary system. No non-public information was used.

---

## Architecture

```
  ┌─────────────────────────────────────────────────────────┐
  │  Control loop process  (:9108 Prometheus metrics)       │
  │                                                         │
  │  K8sCollector ──► PolicyEngine ──► AimdTuner            │
  │      │                │                │                │
  │      │           saturation_score  next_replicas /      │
  │      │           (composite [0,1]) next_max_num_seqs    │
  │      │                                 │                │
  │      └─────────────────────────────────▼                │
  │                                  K8sActuator            │
  │                             (cooldown + bounds)         │
  │                                     │                   │
  │                            KubectlCommandRunner         │
  │                          (single subprocess choke pt)   │
  └──────────────────────────────┬──────────────────────────┘
                                 │ kubectl scale / patch
                                 ▼
                        Kubernetes cluster
                       (vLLM Deployment)

  ┌─────────────────────────────┐
  │  API process  (:8080)       │
  │  FastAPI + static dashboard │
  │  /healthz /serving/snapshot │
  │  /deployment/status /ui     │
  └─────────────────────────────┘
```

---

## Quick start

```bash
# Clone and install
git clone <repo-url> kube-ai && cd kube-ai
pip install -e ".[dev]"

# Copy and edit config
cp config.example.yaml config.yaml
# cp .env.example .env && $EDITOR .env

# Run offline (mock vLLM, dry-run, 3 ticks)
python -m controller.main --dry-run true --interval 1 --max-iterations 3

# Run the API dashboard
uvicorn apps.api.main:app --host 127.0.0.1 --port 8080
# Open http://localhost:8080/ui
```

---

## Mock vs real mode

| `vllm_mode` | `exec_mode` | What happens |
|-------------|-------------|--------------|
| `mock` (default) | any | Static fixture data; parsers run on it; no cluster needed |
| `real` | `local` | Scrapes `vllm_metrics_url`; runs kubectl on host |
| `real` | `ssh` | Scrapes URL; kubectl via SSH tunnel |
| `real` | `docker` | Scrapes URL; kubectl via docker exec |

Switch with `vllm_mode: real` in `config.yaml` or `VLLM_MODE=real` env var.

---

## Direction inversion (vs slurm-ai)

High saturation → **SCALE OUT** (add replicas, raise `max_num_seqs`).
Low saturation  → **scale in** (shed replicas, lower `max_num_seqs`).

AIMD: scale-out step is `+1` (additive, conservative); scale-in step is `current // 2` (multiplicative, aggressive).

---

## Further reading

- `CLAUDE.md` — architecture detail, safety invariants, gotchas
- `AGENTS.md` — sub-task map, monkeypatch contract, test ownership
- `docs/DESIGN.md` — component mapping from slurm-ai, saturation formula, two-tunable design
- `docs/RESEARCH_NOTES.md` — vLLM metric names, paper attributions

---

## License

MIT
