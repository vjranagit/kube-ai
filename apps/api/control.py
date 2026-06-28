"""Control-loop process manager.

Manages a background subprocess running ``python -m controller.main``.
All public functions are idempotent and thread-safe via a module-level lock.

RBAC hook: inject an auth dependency into the FastAPI routes in apps/api/main.py
before exposing start/stop publicly — see the TODO comments on each endpoint.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

_lock = threading.Lock()

# Mutable loop state — only modified while holding _lock.
_proc: subprocess.Popen | None = None  # type: ignore[type-arg]
_loop_state: dict = {
    "running": False,
    "pid": None,
    "started_at": None,
    "error": None,
}


def _reap_unlocked() -> None:
    """Poll subprocess exit; update _loop_state in-place.  Caller must hold _lock."""
    global _proc
    if _proc is None:
        return
    rc = _proc.poll()
    if rc is not None:
        _loop_state.update(running=False, pid=None, error=f"exited(rc={rc})")
        _proc = None


def get_status() -> dict:
    """Return a snapshot of loop state (thread-safe copy)."""
    with _lock:
        _reap_unlocked()
        return dict(_loop_state)


def start_loop(config_path: str | None = None) -> dict:
    """Start the controller subprocess; idempotent if already running."""
    global _proc
    with _lock:
        _reap_unlocked()
        if _loop_state["running"]:
            return {"ok": True, "message": "already running", **_loop_state}

        cmd = [sys.executable, "-m", "controller.main"]
        if config_path:
            cmd += ["--config", config_path]

        try:
            _proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
        except Exception as exc:
            _loop_state["error"] = str(exc)
            return {"ok": False, "message": str(exc), **_loop_state}

        _loop_state.update(
            running=True,
            pid=_proc.pid,
            started_at=time.time(),
            error=None,
        )
        return {"ok": True, "message": "started", **_loop_state}


def stop_loop() -> dict:
    """Terminate the controller subprocess; idempotent if not running."""
    global _proc
    with _lock:
        _reap_unlocked()
        if not _loop_state["running"] or _proc is None:
            return {"ok": True, "message": "not running", **_loop_state}

        try:
            _proc.terminate()
            try:
                _proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _proc.kill()
                _proc.wait()
        except Exception as exc:
            _loop_state["error"] = str(exc)
            return {"ok": False, "message": str(exc), **_loop_state}

        _loop_state.update(running=False, pid=None)
        _proc = None
        return {"ok": True, "message": "stopped", **_loop_state}
