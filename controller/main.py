"""kube-ai control loop entry point.

Usage:
    python -m controller.main [--dry-run true|false] [--interval N] [--config PATH]
                              [--max-iterations N]

--max-iterations N  run exactly N ticks then exit 0 (useful for verification / CI).
                    Default 0 = run forever.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
from dataclasses import asdict

from prometheus_client import start_http_server

from controller.actuator.k8s import K8sActuator
from controller.collectors.k8s import K8sCollector
from controller.config import ControllerConfig
from controller.metrics import (
    ACTION_CHANGED,
    CACHE_PRESSURE,
    KV_CACHE_USAGE,
    LATENCY_PRESSURE,
    P95_TTFT,
    QUEUE_PRESSURE,
    READY_REPLICAS,
    REQUESTS_RUNNING,
    REQUESTS_WAITING,
    SATURATION,
    TARGET_MAX_NUM_SEQS,
    TARGET_REPLICAS,
)
from controller.policy.engine import PolicyEngine
from controller.tuner import build_tuner
from controller.types import PolicyDecision

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
LOG = logging.getLogger("kube-ai")


def run(cfg: ControllerConfig, max_iterations: int = 0) -> None:
    try:
        start_http_server(cfg.metrics_port)
    except OSError as exc:
        LOG.error(
            "Cannot start metrics server on port %d: %s — "
            "check for another process using this port.",
            cfg.metrics_port,
            exc,
        )
        sys.exit(1)
    collector = K8sCollector(cfg)
    policy = PolicyEngine(cfg)
    tuner = build_tuner(cfg)
    actuator = K8sActuator(cfg)

    LOG.info(
        "kube-ai started dry_run=%s interval=%s tune_mode=%s vllm_mode=%s metrics_port=%s",
        cfg.dry_run,
        cfg.interval_sec,
        cfg.tune_mode,
        cfg.vllm_mode,
        cfg.metrics_port,
    )

    # --- Graceful shutdown via SIGTERM / SIGINT ---
    _shutdown = threading.Event()

    def _handle_signal(signum: int, frame: object) -> None:
        LOG.info("signal %d received, initiating graceful shutdown", signum)
        _shutdown.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    iteration = 0
    while not _shutdown.is_set():
        iteration += 1

        try:
            # 1. Collect
            snap = collector.snapshot()

            # 2. Score
            sat = policy.saturation_score(snap)

            # 3. Tune
            target_replicas = tuner.next_replicas(actuator.state.current_replicas, sat)
            target_max_num_seqs = tuner.next_max_num_seqs(actuator.state.current_max_num_seqs, sat)

            decision = PolicyDecision(
                target_replicas=target_replicas,
                target_max_num_seqs=target_max_num_seqs,
                saturation=sat,
                reason=(
                    "scale_out"
                    if sat >= cfg.pressure_high
                    else ("scale_in" if sat <= cfg.pressure_low else "hold")
                ),
            )

            # 4. Actuate
            applied = actuator.apply(decision)

            # 5. Set metrics
            SATURATION.set(sat)
            REQUESTS_WAITING.set(snap.requests_waiting)
            REQUESTS_RUNNING.set(snap.requests_running)
            KV_CACHE_USAGE.set(snap.kv_cache_usage_perc)
            TARGET_REPLICAS.set(applied.new_replicas)
            READY_REPLICAS.set(snap.ready_replicas)
            TARGET_MAX_NUM_SEQS.set(applied.new_max_num_seqs)
            ACTION_CHANGED.set(1 if applied.changed else 0)
            P95_TTFT.set(snap.p95_ttft_sec)
            QUEUE_PRESSURE.set(snap.queue_pressure)
            CACHE_PRESSURE.set(snap.cache_pressure)
            LATENCY_PRESSURE.set(snap.latency_pressure)

            LOG.info(
                "tick=%d %s",
                iteration,
                json.dumps(
                    {
                        "snapshot": asdict(snap),
                        "decision": asdict(decision),
                        "applied": asdict(applied),
                    },
                    default=str,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            LOG.error("tick=%d failed, continuing: %s", iteration, exc, exc_info=True)

        # 6. Exit if bounded run; otherwise sleep (interruptible by shutdown signal)
        if max_iterations > 0 and iteration >= max_iterations:
            LOG.info("max_iterations=%d reached, exiting", max_iterations)
            return

        _shutdown.wait(timeout=cfg.interval_sec)

    LOG.info("kube-ai controller shut down cleanly")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kube-ai control loop")
    parser.add_argument("--dry-run", type=str, default=None, help="true|false")
    parser.add_argument("--interval", type=int, default=None, help="seconds between ticks")
    parser.add_argument("--config", type=str, default=None, help="path to config YAML")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="exit after N ticks (0 = run forever)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Config path override before ControllerConfig is instantiated
    if args.config:
        import os

        os.environ["KUBE_AI_CONFIG"] = args.config

    cfg = ControllerConfig()

    if args.dry_run is not None:
        cfg.dry_run = args.dry_run.lower() == "true"
    if args.interval is not None:
        cfg.interval_sec = args.interval

    run(cfg, max_iterations=args.max_iterations)
