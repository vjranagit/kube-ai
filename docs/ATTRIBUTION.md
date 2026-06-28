# Attribution — kube-ai

## Origin and scope

kube-ai is an independent, from-scratch educational and portfolio implementation of a Kubernetes
adaptive control loop for vLLM serving. All code was written from public sources. It is not
affiliated with, derived from, or representative of any proprietary or internal system. No
non-public information was used.

---

## Inspiring papers (public)

### SelfTune: Tuning Cluster Managers (USENIX NSDI 2023)

- Paper: <https://www.usenix.org/conference/nsdi23/presentation/karthikeyan>
- Video: <https://www.youtube.com/watch?v=DfU2yx1XW8A>
- Contribution to kube-ai: the collect → score → tune → actuate control-loop structure and the
  AIMD-first philosophy are directly inspired by this work. The six-step loop structure and the
  idea of continuous feedback without a workload model come from this paper.

### Chiron: Accelerating Long-Context LLM Inference (arXiv 2501.08090)

- Paper: <https://arxiv.org/abs/2501.08090>
- Contribution to kube-ai: the use of `vllm:num_requests_swapped` as a hard override signal —
  any non-zero swapped count indicates active KV-cache preemption, which is more severe than the
  composite saturation score alone. This insight comes from Chiron's treatment of request
  migration and KV pressure.

---

## Architectural blueprint

The sibling project **slurm-ai** (a Slurm adaptive control loop) served as the architectural
blueprint for kube-ai. The component structure (collector → policy → tuner → actuator → metrics),
the single subprocess choke-point pattern, the monkeypatch test discipline, and the config
resolution order (YAML → ENV → hardcoded default) are all carried over from slurm-ai and adapted
for Kubernetes and vLLM.

---

## Key dependency licenses

| Package | License | Notes |
|---------|---------|-------|
| FastAPI / Starlette | MIT | Web framework and ASGI toolkit |
| uvicorn | BSD (3-clause) | ASGI server |
| pydantic | MIT | Config validation and API models |
| prometheus-client | Apache 2.0 | Prometheus metrics instrumentation |
| PyYAML | MIT | YAML config file parsing |
| pytest | MIT | Test framework |
| ruff | MIT | Linter and formatter |
| Chart.js | MIT | Browser-side charting (CDN, not bundled) |

All dependencies are open-source. No proprietary libraries are required.
