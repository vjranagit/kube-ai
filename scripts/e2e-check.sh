#!/usr/bin/env bash
# e2e-check.sh — full smoke test for the kube-ai local sandbox.
#
# Steps:
#   1. Bring up sandbox (kind + k8s + docker compose)
#   2. Set HIGH load → controller should scale OUT (replicas increase)
#   3. Run controller for N iterations with dry_run=false
#   4. Assert replica count increased
#   5. Set LOW load → controller should scale IN
#   6. Run controller for N more iterations
#   7. Assert replica count decreased (or held at min)
#   8. Print PASS / FAIL
#
# Prerequisites: kind cluster and mock-vllm image must be available.
# The controller must be importable: `python -m controller.main` must work.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$ROOT/tmp"
mkdir -p "$TMP"

KUBECTL="${KUBECTL:-kubectl}"
CLUSTER="kind-kube-ai"
NS="kube-ai"
DEP="vllm-server"
NODEPORT_URL="http://localhost:30080"
SANDBOX_CFG="$ROOT/infra/sandbox.config.yaml"
CONTROLLER_LOG="$TMP/controller.log"
RESULT="FAIL"

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

cleanup() {
  log "--- cleanup ---"
  kill "$CTRL_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Helper: get current desired replicas
# ---------------------------------------------------------------------------
get_replicas() {
  "$KUBECTL" --context "$CLUSTER" get deployment "$DEP" \
    --namespace "$NS" \
    -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0"
}

# ---------------------------------------------------------------------------
# 1. Bring up sandbox
# ---------------------------------------------------------------------------
log "=== Step 1: up.sh ==="
"$ROOT/scripts/up.sh" > "$TMP/up.log" 2>&1 &
UP_PID=$!
DEADLINE=$(( $(date +%s) + 300 ))
while kill -0 "$UP_PID" 2>/dev/null; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    log "ERROR: up.sh timed out after 5 minutes. See tmp/up.log"
    exit 1
  fi
  sleep 5
done
wait "$UP_PID" || { log "ERROR: up.sh failed. See tmp/up.log"; exit 1; }
log "Sandbox up."

# ---------------------------------------------------------------------------
# 2. Set HIGH load → should trigger scale-out
# ---------------------------------------------------------------------------
log "=== Step 2: set high load (0.95) ==="
# Brief poll until mock-vllm is reachable on NodePort
DEADLINE=$(( $(date +%s) + 60 ))
until curl -sf "${NODEPORT_URL}/healthz" &>/dev/null; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    log "ERROR: mock-vllm NodePort not reachable after 60s"
    exit 1
  fi
  sleep 2
done
curl -sf -X POST "${NODEPORT_URL}/admin/set-load" \
  -H "Content-Type: application/json" \
  -d '{"level": 0.95}' >> "$TMP/load-gen.log" 2>&1
log "High load set."

INITIAL_REPLICAS="$(get_replicas)"
log "Initial replicas: $INITIAL_REPLICAS"

# ---------------------------------------------------------------------------
# 3. Run controller (background) for N iterations to let it scale out
# ---------------------------------------------------------------------------
N_ITERS=6
log "=== Step 3: run controller for $N_ITERS iterations (dry_run=false) ==="
KUBE_AI_CONFIG="$SANDBOX_CFG" python -m controller.main \
  --dry-run false \
  --interval 5 \
  --max-iterations "$N_ITERS" \
  > "$CONTROLLER_LOG" 2>&1 &
CTRL_PID=$!
log "Controller running (pid=$CTRL_PID, log: $CONTROLLER_LOG) ..."
wait "$CTRL_PID" || { log "ERROR: controller exited with error. See $CONTROLLER_LOG"; exit 1; }
log "Controller finished $N_ITERS iterations."

# ---------------------------------------------------------------------------
# 4. Assert replicas increased
# ---------------------------------------------------------------------------
log "=== Step 4: assert scale-out ==="
# Give Kubernetes a moment to reflect the scale
sleep 5
AFTER_HIGH_REPLICAS="$(get_replicas)"
log "Replicas after high load: $AFTER_HIGH_REPLICAS (was $INITIAL_REPLICAS)"
if [ "$AFTER_HIGH_REPLICAS" -le "$INITIAL_REPLICAS" ]; then
  log "FAIL: replicas did NOT increase under high load ($INITIAL_REPLICAS → $AFTER_HIGH_REPLICAS)"
  cat "$CONTROLLER_LOG"
  exit 1
fi
log "Scale-out OK: $INITIAL_REPLICAS → $AFTER_HIGH_REPLICAS"

# ---------------------------------------------------------------------------
# 5. Set LOW load → should trigger scale-in
# ---------------------------------------------------------------------------
log "=== Step 5: set low load (0.05) ==="
curl -sf -X POST "${NODEPORT_URL}/admin/set-load" \
  -H "Content-Type: application/json" \
  -d '{"level": 0.05}' >> "$TMP/load-gen.log" 2>&1
log "Low load set."

# ---------------------------------------------------------------------------
# 6. Run controller for more iterations
# ---------------------------------------------------------------------------
log "=== Step 6: run controller for $N_ITERS more iterations ==="
KUBE_AI_CONFIG="$SANDBOX_CFG" python -m controller.main \
  --dry-run false \
  --interval 5 \
  --max-iterations "$N_ITERS" \
  >> "$CONTROLLER_LOG" 2>&1 &
CTRL_PID=$!
wait "$CTRL_PID" || { log "ERROR: controller exited with error. See $CONTROLLER_LOG"; exit 1; }
log "Controller finished second batch."

# ---------------------------------------------------------------------------
# 7. Assert replicas decreased (or at min=1)
# ---------------------------------------------------------------------------
log "=== Step 7: assert scale-in ==="
sleep 5
AFTER_LOW_REPLICAS="$(get_replicas)"
log "Replicas after low load: $AFTER_LOW_REPLICAS (was $AFTER_HIGH_REPLICAS)"
if [ "$AFTER_LOW_REPLICAS" -ge "$AFTER_HIGH_REPLICAS" ]; then
  log "FAIL: replicas did NOT decrease under low load ($AFTER_HIGH_REPLICAS → $AFTER_LOW_REPLICAS)"
  cat "$CONTROLLER_LOG"
  exit 1
fi
log "Scale-in OK: $AFTER_HIGH_REPLICAS → $AFTER_LOW_REPLICAS"

# ---------------------------------------------------------------------------
# 8. Result
# ---------------------------------------------------------------------------
RESULT="PASS"
log ""
log "=========================================="
log " e2e-check: $RESULT"
log " initial=$INITIAL_REPLICAS  high=$AFTER_HIGH_REPLICAS  low=$AFTER_LOW_REPLICAS"
log "=========================================="
