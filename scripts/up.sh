#!/usr/bin/env bash
# up.sh — bring up the kube-ai local sandbox.
# Creates the kind cluster (if absent), applies k8s manifests, and starts
# Prometheus + Grafana via docker compose.
# Heavy steps run in background with logs in tmp/*.log per project rules.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$ROOT/tmp"
mkdir -p "$TMP"

KIND="${KIND:-kind}"
KUBECTL="${KUBECTL:-kubectl}"
CLUSTER_NAME="kube-ai"

# Use a project-local kubeconfig to avoid conflicts with ~/.kube/config being a directory.
export KUBECONFIG="${KUBECONFIG:-$TMP/kubeconfig}"
log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

log "Using KUBECONFIG=$KUBECONFIG"

# ---------------------------------------------------------------------------
# 0. Build mock-vllm image (always rebuild to ensure fresh image)
# ---------------------------------------------------------------------------
log "Building mock-vllm:latest (log: tmp/docker-build.log) ..."
docker build -t mock-vllm:latest "$ROOT/infra/mock-vllm" \
  > "$TMP/docker-build.log" 2>&1
log "mock-vllm:latest built."

# ---------------------------------------------------------------------------
# 1. Create kind cluster (skip if it already exists)
# ---------------------------------------------------------------------------
if "$KIND" get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
  log "kind cluster '$CLUSTER_NAME' already exists — skipping create"
else
  log "Creating kind cluster '$CLUSTER_NAME' (log: tmp/kind-create.log) ..."
  "$KIND" create cluster \
    --name "$CLUSTER_NAME" \
    --config "$ROOT/infra/kind/cluster.yaml" \
    --kubeconfig "$KUBECONFIG" \
    > "$TMP/kind-create.log" 2>&1 &
  KIND_PID=$!
  log "Waiting for kind cluster creation (pid=$KIND_PID) ..."
  # Poll until the background job finishes (max 3 minutes)
  DEADLINE=$(( $(date +%s) + 180 ))
  while kill -0 "$KIND_PID" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
      log "ERROR: kind create timed out after 3 minutes. See tmp/kind-create.log"
      exit 1
    fi
    sleep 3
  done
  wait "$KIND_PID" || { log "ERROR: kind create failed. See tmp/kind-create.log"; exit 1; }
  log "kind cluster created."
fi

# ---------------------------------------------------------------------------
# 2. Load mock-vllm image into kind
# ---------------------------------------------------------------------------
log "Loading mock-vllm:latest into kind cluster (log: tmp/kind-load.log) ..."
"$KIND" load docker-image mock-vllm:latest --name "$CLUSTER_NAME" \
  > "$TMP/kind-load.log" 2>&1
log "mock-vllm:latest loaded into kind."

# ---------------------------------------------------------------------------
# 3. Apply Kubernetes manifests
# ---------------------------------------------------------------------------
log "Applying k8s manifests ..."
MANIFESTS=(
  "$ROOT/infra/k8s/namespace.yaml"
  "$ROOT/infra/k8s/mock-vllm-deployment.yaml"
  "$ROOT/infra/k8s/mock-vllm-service.yaml"
  "$ROOT/infra/k8s/controller-rbac.yaml"
)
for f in "${MANIFESTS[@]}"; do
  KUBECONFIG="$KUBECONFIG" "$KUBECTL" --context "kind-$CLUSTER_NAME" apply -f "$f"
done

# ---------------------------------------------------------------------------
# 4. Start Prometheus + Grafana via docker compose
# ---------------------------------------------------------------------------
COMPOSE_FILE="$ROOT/infra/docker/docker-compose.yml"
log "Starting Prometheus + Grafana (log: tmp/compose-up.log) ..."
docker compose -f "$COMPOSE_FILE" up -d \
  > "$TMP/compose-up.log" 2>&1 &
COMPOSE_PID=$!
DEADLINE=$(( $(date +%s) + 60 ))
while kill -0 "$COMPOSE_PID" 2>/dev/null; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    log "ERROR: docker compose up timed out. See tmp/compose-up.log"
    exit 1
  fi
  sleep 2
done
wait "$COMPOSE_PID" || { log "ERROR: docker compose up failed. See tmp/compose-up.log"; exit 1; }

# ---------------------------------------------------------------------------
# 5. Wait for mock-vllm rollout
# ---------------------------------------------------------------------------
log "Waiting for vllm-server rollout ..."
KUBECONFIG="$KUBECONFIG" "$KUBECTL" --context "kind-$CLUSTER_NAME" \
  rollout status deployment/vllm-server \
  --namespace kube-ai \
  --timeout=120s

log ""
log "Sandbox is up."
log "  Prometheus:  http://localhost:9090"
log "  Grafana:     http://localhost:3000  (admin/admin)"
log "  mock-vllm:   http://localhost:30080/metrics"
log ""
log "To run the controller against the sandbox:"
log "  cp infra/sandbox.config.yaml config.yaml"
log "  python -m controller.main --dry-run false --interval 10"
