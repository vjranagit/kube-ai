#!/usr/bin/env python3
"""Concurrent batch / stress load generator for a real OpenAI-compatible vLLM endpoint.

Fires real /v1/completions requests at escalating concurrency tiers to build genuine
queue pressure (num_requests_waiting / running) on the served model, so the kube-ai
control loop can be exercised against a real workload — not a mock.

Stdlib only (urllib + threading), so it runs anywhere with no extra deps.

Usage:
    python scripts/load-real.py --base-url http://localhost:30080 --model facebook/opt-125m \
        --tiers 2,8,20,50 --secs 45 --max-tokens 64

Each tier runs `secs` seconds with the given number of concurrent workers, then prints
throughput / latency / success stats. Point kube-ai at the same endpoint's /metrics and
watch saturation, replicas, and max_num_seqs react tier by tier.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from dataclasses import dataclass, field


@dataclass
class TierStats:
    sent: int = 0
    ok: int = 0
    err: int = 0
    latencies: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, ok: bool, latency: float) -> None:
        with self._lock:
            self.sent += 1
            if ok:
                self.ok += 1
                self.latencies.append(latency)
            else:
                self.err += 1


def _one_request(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> tuple[bool, float]:
    body = json.dumps(
        {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.7}
    ).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            return resp.status == 200, time.monotonic() - start
    except Exception:
        return False, time.monotonic() - start


def _worker(stop_at: float, base_url: str, model: str, prompt: str, max_tokens: int, stats: TierStats) -> None:
    while time.monotonic() < stop_at:
        ok, lat = _one_request(base_url, model, prompt, max_tokens, timeout=120.0)
        stats.record(ok, lat)


def run_tier(base_url: str, model: str, concurrency: int, secs: int, max_tokens: int, prompt: str) -> TierStats:
    stats = TierStats()
    stop_at = time.monotonic() + secs
    threads = [
        threading.Thread(
            target=_worker, args=(stop_at, base_url, model, prompt, max_tokens, stats), daemon=True
        )
        for _ in range(concurrency)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return stats


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, int(p / 100.0 * len(s)))
    return s[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:30080")
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--tiers", default="2,8,20,50", help="comma-separated concurrency levels")
    ap.add_argument("--secs", type=int, default=45, help="seconds per tier")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt", default="Summarize the benefits of Kubernetes autoscaling in two sentences.")
    args = ap.parse_args()

    tiers = [int(x) for x in args.tiers.split(",") if x.strip()]
    print(f"# load-real -> {args.base_url} model={args.model} tiers={tiers} secs/tier={args.secs}")
    print(f"{'tier':>6} {'sent':>7} {'ok':>7} {'err':>6} {'rps':>8} {'p50_s':>8} {'p95_s':>8}")
    for c in tiers:
        st = run_tier(args.base_url, args.model, c, args.secs, args.max_tokens, args.prompt)
        rps = st.ok / max(1, args.secs)
        print(
            f"{c:>6} {st.sent:>7} {st.ok:>7} {st.err:>6} {rps:>8.2f} "
            f"{_pct(st.latencies, 50):>8.2f} {_pct(st.latencies, 95):>8.2f}"
        )
    print("# done. Watch kube-ai saturation/replicas react across the tiers.")


if __name__ == "__main__":
    main()
