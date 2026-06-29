# Attribution — kube-ai

kube-ai is an independent, open-source implementation of a Kubernetes adaptive control loop for
vLLM / GPU serving. It is not affiliated with, derived from, or representative of any proprietary or
internal system, and uses no non-public information.

## References (public research)

- **SelfTune: Tuning Cluster Managers** — USENIX NSDI 2023
  (<https://www.usenix.org/conference/nsdi23/presentation/karthikeyan>). Informed the
  `collect → score → tune → actuate` control loop and the AIMD-first approach.
- **Chiron** — arXiv 2501.08090 (<https://arxiv.org/abs/2501.08090>). Informed treating any non-zero
  `vllm:num_requests_swapped` (KV-cache preemption) as a hard scale-out signal.

## Dependencies

All open-source: FastAPI / Starlette (MIT), uvicorn (BSD-3-Clause), pydantic (MIT),
prometheus-client (Apache-2.0), PyYAML (MIT), pytest (MIT), ruff (MIT), Chart.js (MIT, via CDN).
No proprietary libraries are required.
