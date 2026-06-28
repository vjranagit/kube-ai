#!/usr/bin/env bash
# down.sh — tear down the kube-ai local sandbox.
# Stops docker compose services and deletes the kind cluster.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$ROOT/tmp"
mkdir -p "$TMP"

KIND="${KIND:-kind}"
CLUSTER_NAME="kube-ai"
COMPOSE_FILE="$ROOT/infra/docker/docker-compose.yml"

# Use project-local kubeconfig
export KUBECONFIG="${KUBECONFIG:-$TMP/kubeconfig}"

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

# ---------------------------------------------------------------------------
# 1. Stop Prometheus + Grafana
# ---------------------------------------------------------------------------
if [ -f "$COMPOSE_FILE" ]; then
  log "Stopping docker compose services ..."
  docker compose -f "$COMPOSE_FILE" down -v \
    > "$TMP/compose-down.log" 2>&1 && log "docker compose services stopped." \
    || log "WARNING: docker compose down failed (see tmp/compose-down.log)"
fi

# ---------------------------------------------------------------------------
# 2. Delete kind cluster
# ---------------------------------------------------------------------------
if "$KIND" get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
  log "Deleting kind cluster '$CLUSTER_NAME' ..."
  "$KIND" delete cluster --name "$CLUSTER_NAME" \
    > "$TMP/kind-delete.log" 2>&1 && log "kind cluster deleted." \
    || { log "ERROR: kind delete failed. See tmp/kind-delete.log"; exit 1; }
else
  log "kind cluster '$CLUSTER_NAME' not found — nothing to delete."
fi

log "Sandbox is down."
