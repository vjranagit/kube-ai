"""Tests for actuator/k8s.py — apply(), dry_run, cooldown, bounds, changed flag, safety."""
from __future__ import annotations

import json
import shlex
from datetime import datetime, timedelta, timezone

import pytest

from controller.actuator.k8s import K8sActuator
from controller.tests.conftest import make_cfg
from controller.types import PolicyDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_decision(
    target_replicas: int = 4,
    target_max_num_seqs: int = 512,
    saturation: float = 0.8,
    reason: str = "scale_out",
) -> PolicyDecision:
    return PolicyDecision(
        target_replicas=target_replicas,
        target_max_num_seqs=target_max_num_seqs,
        saturation=saturation,
        reason=reason,
    )


def expire_cooldowns(actuator: K8sActuator) -> None:
    """Push last_*_apply far into the past so cooldowns are always expired."""
    past = datetime.now(timezone.utc) - timedelta(hours=24)
    actuator.last_replica_apply = past
    actuator.last_param_apply = past


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_initial_replicas_within_bounds() -> None:
    cfg = make_cfg(min_replicas=1, max_replicas=8)
    actuator = K8sActuator(cfg)
    assert cfg.min_replicas <= actuator.state.current_replicas <= cfg.max_replicas


def test_initial_max_num_seqs_equals_min_max_num_seqs() -> None:
    cfg = make_cfg(min_max_num_seqs=128, max_max_num_seqs=2048)
    actuator = K8sActuator(cfg)
    assert actuator.state.current_max_num_seqs == 128


def test_initial_last_replica_apply_is_min_datetime() -> None:
    actuator = K8sActuator(make_cfg())
    assert actuator.last_replica_apply == datetime.min.replace(tzinfo=timezone.utc)


def test_initial_last_param_apply_is_min_datetime() -> None:
    actuator = K8sActuator(make_cfg())
    assert actuator.last_param_apply == datetime.min.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# dry_run=True: no runner calls, log starts with DRY_RUN
# ---------------------------------------------------------------------------


def test_dry_run_makes_no_runner_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    actuator = K8sActuator(make_cfg(dry_run=True))
    called: list[str] = []

    def fake_run(command: str, check: bool = True) -> tuple[bool, str]:
        called.append(command)
        return True, ""

    monkeypatch.setattr(actuator.runner, "run", fake_run)
    actuator.apply(make_decision())
    assert called == []


def test_dry_run_log_starts_with_dry_run() -> None:
    actuator = K8sActuator(make_cfg(dry_run=True, tune_mode="both"))
    action = actuator.apply(make_decision())
    for entry in action.command_log:
        assert entry.startswith("DRY_RUN"), f"Expected DRY_RUN prefix, got: {entry!r}"


def test_dry_run_log_is_non_empty() -> None:
    actuator = K8sActuator(make_cfg(dry_run=True))
    action = actuator.apply(make_decision())
    assert len(action.command_log) > 0


def test_dry_run_state_advances() -> None:
    """dry_run still simulates state advance (behavior is intentional)."""
    actuator = K8sActuator(make_cfg(dry_run=True, min_replicas=1, max_replicas=8))
    action = actuator.apply(make_decision(target_replicas=5))
    assert actuator.state.current_replicas == 5
    assert action.new_replicas == 5


# ---------------------------------------------------------------------------
# Live mode: runner called, log starts with OK or ERR
# ---------------------------------------------------------------------------


def test_live_mode_calls_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    actuator = K8sActuator(make_cfg(dry_run=False, tune_mode="both"))
    called: list[str] = []

    def fake_run(command: str, check: bool = True) -> tuple[bool, str]:
        called.append(command)
        return True, "ok"

    monkeypatch.setattr(actuator.runner, "run", fake_run)
    actuator.apply(make_decision())
    assert len(called) >= 1


def test_live_mode_ok_log_starts_with_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    actuator = K8sActuator(make_cfg(dry_run=False, tune_mode="both"))
    monkeypatch.setattr(actuator.runner, "run", lambda cmd, check=True: (True, "done"))
    action = actuator.apply(make_decision())
    assert all(e.startswith("OK") or "cooldown" in e for e in action.command_log)


def test_live_mode_err_log_starts_with_err(monkeypatch: pytest.MonkeyPatch) -> None:
    actuator = K8sActuator(make_cfg(dry_run=False, tune_mode="replicas"))
    monkeypatch.setattr(actuator.runner, "run", lambda cmd, check=True: (False, "error"))
    action = actuator.apply(make_decision())
    assert any(e.startswith("ERR") for e in action.command_log)


def test_live_mode_scale_command_contains_replicas_subcommand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actuator = K8sActuator(make_cfg(dry_run=False, tune_mode="replicas"))
    called: list[str] = []
    monkeypatch.setattr(actuator.runner, "run", lambda cmd, check=True: called.append(cmd) or (True, ""))
    actuator.apply(make_decision(target_replicas=3))
    assert any("scale" in c for c in called)


def test_live_mode_patch_command_contains_patch_subcommand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actuator = K8sActuator(make_cfg(dry_run=False, tune_mode="params"))
    called: list[str] = []
    monkeypatch.setattr(actuator.runner, "run", lambda cmd, check=True: called.append(cmd) or (True, ""))
    actuator.apply(make_decision(target_max_num_seqs=512))
    assert any("patch" in c for c in called)


# ---------------------------------------------------------------------------
# TUNE_MODE gating
# ---------------------------------------------------------------------------


def test_tune_mode_replicas_only_no_patch_log() -> None:
    actuator = K8sActuator(make_cfg(tune_mode="replicas"))
    action = actuator.apply(make_decision())
    assert not any("patch" in e for e in action.command_log)


def test_tune_mode_replicas_only_has_scale_log() -> None:
    actuator = K8sActuator(make_cfg(tune_mode="replicas"))
    action = actuator.apply(make_decision())
    assert any("scale" in e for e in action.command_log)


def test_tune_mode_params_only_no_scale_log() -> None:
    actuator = K8sActuator(make_cfg(tune_mode="params"))
    action = actuator.apply(make_decision())
    assert not any("scale" in e for e in action.command_log)


def test_tune_mode_params_only_has_patch_log() -> None:
    actuator = K8sActuator(make_cfg(tune_mode="params"))
    action = actuator.apply(make_decision())
    assert any("patch" in e for e in action.command_log)


def test_tune_mode_both_has_scale_and_patch_log() -> None:
    actuator = K8sActuator(make_cfg(tune_mode="both"))
    action = actuator.apply(make_decision())
    assert any("scale" in e for e in action.command_log)
    assert any("patch" in e for e in action.command_log)


def test_tune_mode_params_replica_state_unchanged() -> None:
    actuator = K8sActuator(make_cfg(tune_mode="params"))
    initial_replicas = actuator.state.current_replicas
    actuator.apply(make_decision(target_replicas=7))
    assert actuator.state.current_replicas == initial_replicas


def test_tune_mode_replicas_param_state_unchanged() -> None:
    actuator = K8sActuator(make_cfg(tune_mode="replicas"))
    initial_seqs = actuator.state.current_max_num_seqs
    actuator.apply(make_decision(target_max_num_seqs=1024))
    assert actuator.state.current_max_num_seqs == initial_seqs


# ---------------------------------------------------------------------------
# Double-clamping beyond bounds
# ---------------------------------------------------------------------------


def test_clamp_target_replicas_above_max() -> None:
    cfg = make_cfg(max_replicas=8, tune_mode="replicas")
    actuator = K8sActuator(cfg)
    action = actuator.apply(make_decision(target_replicas=999))
    assert action.new_replicas == 8


def test_clamp_target_replicas_below_min() -> None:
    cfg = make_cfg(min_replicas=2, tune_mode="replicas")
    actuator = K8sActuator(cfg)
    action = actuator.apply(make_decision(target_replicas=0))
    assert action.new_replicas == 2


def test_clamp_target_max_num_seqs_above_max() -> None:
    cfg = make_cfg(max_max_num_seqs=2048, tune_mode="params")
    actuator = K8sActuator(cfg)
    action = actuator.apply(make_decision(target_max_num_seqs=99999))
    assert action.new_max_num_seqs == 2048


def test_clamp_target_max_num_seqs_below_min() -> None:
    cfg = make_cfg(min_max_num_seqs=128, tune_mode="params")
    actuator = K8sActuator(cfg)
    action = actuator.apply(make_decision(target_max_num_seqs=1))
    assert action.new_max_num_seqs == 128


# ---------------------------------------------------------------------------
# Changed flag
# ---------------------------------------------------------------------------


def test_changed_true_when_replicas_change() -> None:
    actuator = K8sActuator(make_cfg(tune_mode="replicas"))
    initial = actuator.state.current_replicas
    action = actuator.apply(make_decision(target_replicas=initial + 2))
    assert action.changed is True


def test_changed_false_when_nothing_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    actuator = K8sActuator(make_cfg(tune_mode="both", cooldown_sec=0, param_cooldown_sec=0))
    # First apply to set state
    actuator.apply(make_decision(target_replicas=4, target_max_num_seqs=512))
    expire_cooldowns(actuator)
    # Same values again → no change
    action = actuator.apply(make_decision(target_replicas=4, target_max_num_seqs=512))
    assert action.changed is False


def test_changed_true_when_only_seqs_change(monkeypatch: pytest.MonkeyPatch) -> None:
    actuator = K8sActuator(make_cfg(tune_mode="params", cooldown_sec=0, param_cooldown_sec=0))
    actuator.apply(make_decision(target_max_num_seqs=256))
    expire_cooldowns(actuator)
    action = actuator.apply(make_decision(target_max_num_seqs=512))
    assert action.changed is True


# ---------------------------------------------------------------------------
# Replica cooldown
# ---------------------------------------------------------------------------


def test_replica_cooldown_blocks_second_apply() -> None:
    actuator = K8sActuator(make_cfg(dry_run=True, cooldown_sec=3600, tune_mode="replicas"))
    actuator.apply(make_decision(target_replicas=2))
    # Simulate last_replica_apply just now
    actuator.last_replica_apply = datetime.now(timezone.utc)
    action = actuator.apply(make_decision(target_replicas=4))
    assert any("cooldown" in e for e in action.command_log)
    assert action.changed is False


def test_replica_cooldown_does_not_block_after_expiry() -> None:
    actuator = K8sActuator(make_cfg(dry_run=True, cooldown_sec=10, tune_mode="replicas"))
    actuator.apply(make_decision(target_replicas=2))
    actuator.last_replica_apply = datetime.now(timezone.utc) - timedelta(seconds=11)
    action = actuator.apply(make_decision(target_replicas=5))
    assert not all("cooldown" in e for e in action.command_log)


# ---------------------------------------------------------------------------
# Param cooldown
# ---------------------------------------------------------------------------


def test_param_cooldown_blocks_second_apply() -> None:
    actuator = K8sActuator(make_cfg(dry_run=True, param_cooldown_sec=3600, tune_mode="params"))
    actuator.apply(make_decision(target_max_num_seqs=256))
    actuator.last_param_apply = datetime.now(timezone.utc)
    action = actuator.apply(make_decision(target_max_num_seqs=512))
    assert any("cooldown" in e for e in action.command_log)
    assert action.changed is False


def test_param_cooldown_replica_cooldown_are_independent() -> None:
    """Replica cooldown should not block param path and vice versa."""
    cfg = make_cfg(dry_run=True, tune_mode="both", cooldown_sec=3600, param_cooldown_sec=0)
    actuator = K8sActuator(cfg)
    actuator.apply(make_decision(target_replicas=2, target_max_num_seqs=256))
    # Block replica path only
    actuator.last_replica_apply = datetime.now(timezone.utc)
    # Param path should still proceed (cooldown=0 so always expires instantly)
    action = actuator.apply(make_decision(target_replicas=6, target_max_num_seqs=512))
    # Replicas blocked, params may change
    assert any("replicas" in e and "cooldown" in e for e in action.command_log)


# ---------------------------------------------------------------------------
# Key safety regression: state advances ONLY on ok=True
# ---------------------------------------------------------------------------


def test_live_command_failure_replica_state_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """State must not advance when kubectl scale returns (False, err)."""
    cfg = make_cfg(dry_run=False, tune_mode="replicas")
    actuator = K8sActuator(cfg)
    initial_replicas = actuator.state.current_replicas

    monkeypatch.setattr(actuator.runner, "run", lambda cmd, check=True: (False, "permission denied"))
    action = actuator.apply(make_decision(target_replicas=6))

    assert actuator.state.current_replicas == initial_replicas
    assert action.changed is False
    assert action.new_replicas == initial_replicas


def test_live_command_failure_param_state_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """State must not advance when kubectl patch returns (False, err)."""
    cfg = make_cfg(dry_run=False, tune_mode="params")
    actuator = K8sActuator(cfg)
    initial_seqs = actuator.state.current_max_num_seqs

    monkeypatch.setattr(actuator.runner, "run", lambda cmd, check=True: (False, "forbidden"))
    action = actuator.apply(make_decision(target_max_num_seqs=1024))

    assert actuator.state.current_max_num_seqs == initial_seqs
    assert action.changed is False
    assert action.new_max_num_seqs == initial_seqs


def test_live_command_failure_does_not_update_last_replica_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_cfg(dry_run=False, tune_mode="replicas")
    actuator = K8sActuator(cfg)
    original_last = actuator.last_replica_apply

    monkeypatch.setattr(actuator.runner, "run", lambda cmd, check=True: (False, "err"))
    actuator.apply(make_decision(target_replicas=4))

    assert actuator.last_replica_apply == original_last


def test_live_command_failure_does_not_update_last_param_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_cfg(dry_run=False, tune_mode="params")
    actuator = K8sActuator(cfg)
    original_last = actuator.last_param_apply

    monkeypatch.setattr(actuator.runner, "run", lambda cmd, check=True: (False, "err"))
    actuator.apply(make_decision(target_max_num_seqs=512))

    assert actuator.last_param_apply == original_last


def test_live_command_success_does_advance_last_replica_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_cfg(dry_run=False, tune_mode="replicas")
    actuator = K8sActuator(cfg)
    original_last = actuator.last_replica_apply

    monkeypatch.setattr(actuator.runner, "run", lambda cmd, check=True: (True, "ok"))
    actuator.apply(make_decision(target_replicas=4))

    assert actuator.last_replica_apply > original_last


# ---------------------------------------------------------------------------
# old/new fields in AppliedAction
# ---------------------------------------------------------------------------


def test_action_old_replicas_reflects_state_before_apply() -> None:
    actuator = K8sActuator(make_cfg())
    initial = actuator.state.current_replicas
    action = actuator.apply(make_decision(target_replicas=initial + 2))
    assert action.old_replicas == initial


def test_action_old_max_num_seqs_reflects_state_before_apply() -> None:
    actuator = K8sActuator(make_cfg())
    initial = actuator.state.current_max_num_seqs
    action = actuator.apply(make_decision(target_max_num_seqs=initial + 128))
    assert action.old_max_num_seqs == initial


# ---------------------------------------------------------------------------
# _sync_initial_state is best-effort: failures must not raise
# ---------------------------------------------------------------------------


def test_sync_initial_state_runner_failure_keeps_clamped_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_sync_initial_state() keeps the safe default when the runner returns failure."""
    from controller.kubectl_exec import KubectlCommandRunner

    monkeypatch.setattr(
        KubectlCommandRunner,
        "run",
        lambda self, cmd, check=True: (False, "connection refused"),
    )
    cfg = make_cfg(min_replicas=1, max_replicas=8)
    actuator = K8sActuator(cfg)  # must not raise
    assert actuator.state.current_replicas == 1  # clamped default unchanged


def test_sync_initial_state_runner_exception_does_not_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_sync_initial_state() swallows unexpected runner exceptions (best-effort)."""
    from controller.kubectl_exec import KubectlCommandRunner

    def _explode(self: object, cmd: str, check: bool = True) -> None:
        raise RuntimeError("unexpected error during sync")

    monkeypatch.setattr(KubectlCommandRunner, "run", _explode)
    cfg = make_cfg(min_replicas=2, max_replicas=8)
    actuator = K8sActuator(cfg)  # must not raise
    assert cfg.min_replicas <= actuator.state.current_replicas <= cfg.max_replicas


# ---------------------------------------------------------------------------
# C2 — Shell injection: deployment/namespace names must be shlex-quoted
# ---------------------------------------------------------------------------


def test_scale_cmd_quotes_metacharacter_deployment_name() -> None:
    """Metacharacters in vllm_deployment are shlex-quoted in _scale_cmd (C2).

    The shell-quoted form of the dangerous name must appear in the command so
    that the shell treats it as a single argument, not as multiple tokens.
    """
    dangerous = "vllm; rm -rf /tmp/x"
    cfg = make_cfg(vllm_deployment=dangerous, vllm_namespace="default")
    actuator = K8sActuator(cfg)
    cmd = actuator._scale_cmd(3)  # noqa: SLF001
    # The properly shell-quoted form must be present (proves safe quoting)
    assert shlex.quote(dangerous) in cmd


def test_patch_cmd_quotes_metacharacter_namespace() -> None:
    """Metacharacters in vllm_namespace are shlex-quoted in _patch_cmd (C2)."""
    dangerous_ns = "default && cat /etc/passwd"
    cfg = make_cfg(vllm_deployment="vllm-server", vllm_namespace=dangerous_ns)
    actuator = K8sActuator(cfg)
    cmd = actuator._patch_cmd(256, [])  # noqa: SLF001
    # Shell-quoted form present → namespace is a single token, not split on &&
    assert shlex.quote(dangerous_ns) in cmd


# ---------------------------------------------------------------------------
# C4 — Args-preserving patch: only --max-num-seqs changes, others survive
# ---------------------------------------------------------------------------


def _decode_patch_args(cmd: str) -> list[str]:
    """Extract and decode the container args from a _patch_cmd output string."""
    # The command is: ... -p <json-string-of-json>
    p_idx = cmd.rindex(" -p ")
    p_value = cmd[p_idx + 4:]
    # Double-decode: outer JSON string → inner JSON object
    patch_dict = json.loads(json.loads(p_value))
    containers = patch_dict["spec"]["template"]["spec"]["containers"]
    return containers[0]["args"]


def test_patch_cmd_preserves_existing_args() -> None:
    """_patch_cmd updates only --max-num-seqs; all other args are preserved (C4)."""
    current_args = [
        "--model=/data/models/mistral",
        "--dtype=float16",
        "--max-num-seqs=128",
        "--tensor-parallel-size=2",
    ]
    cfg = make_cfg()
    actuator = K8sActuator(cfg)
    cmd = actuator._patch_cmd(256, current_args)  # noqa: SLF001
    args = _decode_patch_args(cmd)
    assert "--model=/data/models/mistral" in args
    assert "--dtype=float16" in args
    assert "--tensor-parallel-size=2" in args
    assert "--max-num-seqs=256" in args
    assert "--max-num-seqs=128" not in args


def test_patch_cmd_appends_max_num_seqs_when_absent() -> None:
    """_patch_cmd appends --max-num-seqs when the arg is not yet present (C4)."""
    current_args = ["--model=/data/models/mistral", "--dtype=bfloat16"]
    cfg = make_cfg()
    actuator = K8sActuator(cfg)
    cmd = actuator._patch_cmd(512, current_args)  # noqa: SLF001
    args = _decode_patch_args(cmd)
    assert "--model=/data/models/mistral" in args
    assert "--max-num-seqs=512" in args


def test_patch_cmd_live_args_fetched_in_live_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """In live (non-dry-run) mode, apply() fetches live args before patching (C4)."""
    import json as _json

    live_args = ["--model=mistral", "--dtype=float16", "--max-num-seqs=128"]
    dep_json = _json.dumps({
        "spec": {"template": {"spec": {"containers": [
            {"name": "vllm-server", "args": live_args}
        ]}}}
    })

    cfg = make_cfg(dry_run=False, tune_mode="params")
    actuator = K8sActuator(cfg)
    # State: seqs at 128, target at 256 → a change will be issued
    actuator.state.current_max_num_seqs = 128
    expire_cooldowns(actuator)

    # Runner returns dep JSON for get, then success for patch
    responses = [(True, dep_json), (True, "patched")]
    call_idx = [0]
    calls: list[str] = []

    def fake_run(cmd: str, check: bool = True) -> tuple[bool, str]:
        calls.append(cmd)
        resp = responses[min(call_idx[0], len(responses) - 1)]
        call_idx[0] += 1
        return resp

    monkeypatch.setattr(actuator.runner, "run", fake_run)
    actuator.apply(make_decision(target_max_num_seqs=256))

    patch_calls = [c for c in calls if "patch" in c]
    assert len(patch_calls) == 1
    args = _decode_patch_args(patch_calls[0])
    assert "--model=mistral" in args
    assert "--dtype=float16" in args
    assert "--max-num-seqs=256" in args
    assert "--max-num-seqs=128" not in args


# ---------------------------------------------------------------------------
# H3 — _sync_initial_state also syncs max_num_seqs from container args
# ---------------------------------------------------------------------------


def test_sync_initial_state_syncs_max_num_seqs(monkeypatch: pytest.MonkeyPatch) -> None:
    """_sync_initial_state reads --max-num-seqs from live container args (H3)."""
    dep_json = json.dumps({
        "spec": {
            "replicas": 3,
            "template": {"spec": {"containers": [
                {"name": "vllm-server", "args": ["--max-num-seqs=512", "--model=mistral"]}
            ]}},
        },
        "status": {},
    })
    from controller.kubectl_exec import KubectlCommandRunner

    monkeypatch.setattr(
        KubectlCommandRunner, "run",
        lambda self, cmd, check=True: (True, dep_json),
    )
    cfg = make_cfg(min_replicas=1, max_replicas=8, min_max_num_seqs=128, max_max_num_seqs=2048)
    actuator = K8sActuator(cfg)
    assert actuator.state.current_replicas == 3
    assert actuator.state.current_max_num_seqs == 512


def test_sync_initial_state_clamps_max_num_seqs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synced max_num_seqs is clamped to [min, max] bounds (H3)."""
    dep_json = json.dumps({
        "spec": {
            "replicas": 2,
            "template": {"spec": {"containers": [
                {"name": "vllm-server", "args": ["--max-num-seqs=99999"]}
            ]}},
        }
    })
    from controller.kubectl_exec import KubectlCommandRunner

    monkeypatch.setattr(
        KubectlCommandRunner, "run",
        lambda self, cmd, check=True: (True, dep_json),
    )
    cfg = make_cfg(min_max_num_seqs=128, max_max_num_seqs=1024)
    actuator = K8sActuator(cfg)
    assert actuator.state.current_max_num_seqs == 1024  # clamped to max


# ---------------------------------------------------------------------------
# H4 — Idempotency: no kubectl command when target equals current
# ---------------------------------------------------------------------------


def test_idempotency_no_scale_when_replicas_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """No kubectl call is made when target replicas equal current replicas (H4)."""
    cfg = make_cfg(dry_run=False, tune_mode="replicas")
    actuator = K8sActuator(cfg)
    actuator.state.current_replicas = 4
    expire_cooldowns(actuator)

    calls: list[str] = []
    monkeypatch.setattr(actuator.runner, "run", lambda cmd, check=True: calls.append(cmd) or (True, ""))

    action = actuator.apply(make_decision(target_replicas=4))
    assert not any("scale" in c for c in calls), "scale command should not be issued when no change"
    assert action.changed is False


def test_idempotency_no_patch_when_seqs_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """No kubectl call is made when target max_num_seqs equals current (H4)."""
    cfg = make_cfg(dry_run=False, tune_mode="params")
    actuator = K8sActuator(cfg)
    actuator.state.current_max_num_seqs = 512
    expire_cooldowns(actuator)

    calls: list[str] = []
    monkeypatch.setattr(actuator.runner, "run", lambda cmd, check=True: calls.append(cmd) or (True, ""))

    action = actuator.apply(make_decision(target_max_num_seqs=512))
    assert not any("patch" in c for c in calls), "patch command should not be issued when no change"
    assert action.changed is False


def test_idempotency_command_log_notes_no_change() -> None:
    """'no change' appears in command_log when the target equals current (H4)."""
    cfg = make_cfg(dry_run=True, tune_mode="both")
    actuator = K8sActuator(cfg)
    current_r = actuator.state.current_replicas
    current_s = actuator.state.current_max_num_seqs
    expire_cooldowns(actuator)

    action = actuator.apply(make_decision(target_replicas=current_r, target_max_num_seqs=current_s))
    assert any("no change" in e for e in action.command_log)
