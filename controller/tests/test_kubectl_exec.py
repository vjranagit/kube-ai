"""Tests for kubectl_exec.py — command building, modes, (ok, out) contract."""
from __future__ import annotations

import shutil

import pytest

from controller.kubectl_exec import KubectlCommandRunner, KubectlExecConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_runner(
    mode: str = "local",
    context: str = "",
    namespace: str = "default",
    ssh_host: str = "",
    ssh_user: str = "",
    ssh_key_file: str = "",
    docker_container: str = "",
) -> KubectlCommandRunner:
    return KubectlCommandRunner(
        KubectlExecConfig(
            mode=mode,
            context=context,
            namespace=namespace,
            ssh_host=ssh_host,
            ssh_user=ssh_user,
            ssh_key_file=ssh_key_file,
            docker_container=docker_container,
        )
    )


def build(runner: KubectlCommandRunner, cmd: str) -> str:
    return runner._build(cmd)  # noqa: SLF001


# ---------------------------------------------------------------------------
# Local mode — command structure
# ---------------------------------------------------------------------------


def test_local_mode_starts_with_kubectl() -> None:
    runner = make_runner(mode="local")
    assert build(runner, "get pods").startswith("kubectl")


def test_local_mode_contains_subcommand() -> None:
    runner = make_runner(mode="local")
    assert "get pods" in build(runner, "get pods")


def test_local_mode_with_namespace_includes_namespace_flag() -> None:
    runner = make_runner(mode="local", namespace="production")
    result = build(runner, "get pods")
    assert "--namespace" in result
    assert "production" in result


def test_local_mode_with_context_includes_context_flag() -> None:
    runner = make_runner(mode="local", context="my-cluster")
    result = build(runner, "get pods")
    assert "--context" in result
    assert "my-cluster" in result


def test_local_mode_no_context_omits_context_flag() -> None:
    runner = make_runner(mode="local", context="")
    result = build(runner, "get pods")
    assert "--context" not in result


def test_local_mode_empty_namespace_omits_namespace_flag() -> None:
    runner = make_runner(mode="local", namespace="")
    result = build(runner, "get pods")
    assert "--namespace" not in result


def test_local_mode_context_before_namespace_in_output() -> None:
    runner = make_runner(mode="local", context="ctx", namespace="ns")
    result = build(runner, "get pods")
    assert result.index("--context") < result.index("--namespace")


def test_local_mode_does_not_contain_ssh() -> None:
    runner = make_runner(mode="local")
    assert "ssh" not in build(runner, "get pods")


def test_local_mode_does_not_contain_docker() -> None:
    runner = make_runner(mode="local")
    assert "docker" not in build(runner, "get pods")


# ---------------------------------------------------------------------------
# SSH mode — command structure
# ---------------------------------------------------------------------------


def test_ssh_mode_starts_with_ssh() -> None:
    runner = make_runner(mode="ssh", ssh_host="cluster.example.com")
    assert build(runner, "get pods").startswith("ssh")


def test_ssh_mode_contains_host() -> None:
    runner = make_runner(mode="ssh", ssh_host="cluster.example.com")
    assert "cluster.example.com" in build(runner, "get pods")


def test_ssh_mode_with_user_includes_user_at_host() -> None:
    runner = make_runner(mode="ssh", ssh_host="h.example.com", ssh_user="admin")
    result = build(runner, "get pods")
    assert "admin@h.example.com" in result


def test_ssh_mode_without_user_omits_at_sign_before_host() -> None:
    runner = make_runner(mode="ssh", ssh_host="h.example.com", ssh_user="")
    result = build(runner, "get pods")
    # host appears without user@ prefix (within the shlex-quoted arg)
    assert "admin@" not in result


def test_ssh_mode_with_key_includes_identity_flag() -> None:
    runner = make_runner(mode="ssh", ssh_host="h.example.com", ssh_key_file="/home/u/.ssh/id_rsa")
    result = build(runner, "get pods")
    assert "-i" in result
    assert "/home/u/.ssh/id_rsa" in result


def test_ssh_mode_without_key_omits_identity_flag() -> None:
    runner = make_runner(mode="ssh", ssh_host="h.example.com", ssh_key_file="")
    result = build(runner, "get pods")
    # The -i flag for identity file should be absent
    assert " -i " not in result


def test_ssh_mode_missing_host_raises_value_error() -> None:
    runner = make_runner(mode="ssh", ssh_host="")
    with pytest.raises(ValueError, match="SSH_HOST"):
        build(runner, "get pods")


def test_ssh_mode_contains_strict_host_checking_option() -> None:
    runner = make_runner(mode="ssh", ssh_host="h.example.com")
    result = build(runner, "get pods")
    assert "StrictHostKeyChecking" in result


# ---------------------------------------------------------------------------
# Docker mode — command structure
# ---------------------------------------------------------------------------


def test_docker_mode_starts_with_docker() -> None:
    runner = make_runner(mode="docker", docker_container="kubectl-proxy")
    assert build(runner, "get pods").startswith("docker")


def test_docker_mode_contains_exec_subcommand() -> None:
    runner = make_runner(mode="docker", docker_container="kubectl-proxy")
    assert "exec" in build(runner, "get pods")


def test_docker_mode_contains_container_name() -> None:
    runner = make_runner(mode="docker", docker_container="kubectl-proxy")
    assert "kubectl-proxy" in build(runner, "get pods")


def test_docker_mode_missing_container_raises_value_error() -> None:
    runner = make_runner(mode="docker", docker_container="")
    with pytest.raises(ValueError, match="DOCKER_CONTAINER"):
        build(runner, "get pods")


def test_docker_mode_wraps_kubectl_in_sh_c() -> None:
    runner = make_runner(mode="docker", docker_container="c")
    result = build(runner, "get pods")
    assert "sh -c" in result


def test_docker_mode_does_not_contain_ssh() -> None:
    runner = make_runner(mode="docker", docker_container="c")
    assert "ssh" not in build(runner, "get pods")


# ---------------------------------------------------------------------------
# Unsupported mode
# ---------------------------------------------------------------------------


def test_unsupported_mode_raises_value_error() -> None:
    runner = make_runner(mode="telnet")
    with pytest.raises(ValueError, match="unsupported exec_mode"):
        build(runner, "get pods")


# ---------------------------------------------------------------------------
# (ok, out) contract — via monkeypatched subprocess
# ---------------------------------------------------------------------------


def test_run_returns_true_on_zero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    runner = make_runner()

    class FakeResult:
        returncode = 0
        stdout = "ok output"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())
    ok, out = runner.run("get pods")
    assert ok is True


def test_run_returns_output_string(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    runner = make_runner()

    class FakeResult:
        returncode = 0
        stdout = "pod-name-abc"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())
    ok, out = runner.run("get pods")
    assert out == "pod-name-abc"


def test_run_returns_false_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    runner = make_runner()

    class FakeResult:
        returncode = 1
        stdout = ""
        stderr = "Error from server"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())
    ok, out = runner.run("get pods", check=False)
    assert ok is False


def test_run_returns_false_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    runner = make_runner()

    def raise_timeout(*a: object, **k: object) -> None:
        raise subprocess.TimeoutExpired(cmd="kubectl get pods", timeout=30)

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    ok, out = runner.run("get pods")
    assert ok is False
    assert "timed out" in out


def test_run_includes_timeout_command_in_message(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    runner = make_runner()

    def raise_timeout(*a: object, **k: object) -> None:
        raise subprocess.TimeoutExpired(cmd="kubectl", timeout=30)

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    ok, out = runner.run("scale deployment x --replicas=2")
    assert "scale deployment x --replicas=2" in out


# ---------------------------------------------------------------------------
# run() catches _build() exceptions — never raises, always returns (False, err)
# ---------------------------------------------------------------------------


def test_run_returns_false_when_build_raises() -> None:
    """run() catches ValueError from _build() and returns (False, err) instead of raising.

    SSH mode with no ssh_host causes _build() to raise ValueError.  The "Never raises"
    contract on run() must hold even in that case.
    """
    runner = make_runner(mode="ssh", ssh_host="")
    ok, out = runner.run("get pods")
    assert ok is False
    assert out != ""


def test_run_error_message_contains_detail_when_build_raises() -> None:
    """Error string from _build() exception is surfaced in the returned output."""
    runner = make_runner(mode="docker", docker_container="")
    ok, out = runner.run("get pods")
    assert ok is False
    assert "DOCKER_CONTAINER" in out


# ---------------------------------------------------------------------------
# Live kubectl check — skipped unless kubectl binary present
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("kubectl") is None, reason="kubectl not installed")
def test_kubectl_version_client_returns_ok() -> None:
    runner = make_runner(mode="local", namespace="")
    ok, out = runner.run("version --client")
    assert ok is True
    assert out != ""
