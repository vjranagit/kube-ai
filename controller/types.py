from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class ServingSnapshot:
    """Point-in-time view of a vLLM serving deployment."""

    timestamp: datetime

    # Kubernetes deployment state
    desired_replicas: int
    ready_replicas: int
    available_replicas: int

    # vLLM request counters
    requests_waiting: int   # vllm:num_requests_waiting
    requests_running: int   # vllm:num_requests_running
    requests_swapped: int   # vllm:num_requests_swapped (non-zero = KV preemption)

    # vLLM resource signals
    kv_cache_usage_perc: float  # vllm:kv_cache_usage_perc  [0..1]

    # Latency signals
    p95_ttft_sec: float  # approximated from vllm:time_to_first_token_seconds histogram
    p50_ttft_sec: float

    # Composite sub-scores [0..1] (set by policy engine)
    queue_pressure: float
    cache_pressure: float
    latency_pressure: float

    # Current vLLM param
    current_max_num_seqs: int

    # False if vLLM metrics scrape failed; saturation is forced to 0.0 when False
    metrics_available: bool


@dataclass(slots=True)
class PolicyDecision:
    target_replicas: int
    target_max_num_seqs: int
    saturation: float
    reason: str


@dataclass(slots=True)
class AppliedAction:
    changed: bool
    old_replicas: int
    new_replicas: int
    old_max_num_seqs: int
    new_max_num_seqs: int
    command_log: list[str] = field(default_factory=list)
