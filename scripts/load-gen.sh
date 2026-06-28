#!/usr/bin/env bash
# load-gen.sh <level> — set synthetic load level on mock-vllm.
# <level> is a float 0.0..1.0. Sends POST /admin/set-load to the NodePort.
#
# Usage:
#   scripts/load-gen.sh 0.9    # high load
#   scripts/load-gen.sh 0.1    # low load
#   scripts/load-gen.sh reset  # resume sawtooth
set -euo pipefail

LEVEL="${1:-}"
NODEPORT_URL="http://localhost:30080"

if [ -z "$LEVEL" ]; then
  echo "Usage: $0 <level|reset>"
  echo "  level : float 0.0..1.0"
  echo "  reset : resume sawtooth wave"
  exit 1
fi

if [ "$LEVEL" = "reset" ]; then
  echo "Resetting mock-vllm load model to sawtooth ..."
  curl -sf -X POST "${NODEPORT_URL}/admin/reset" \
    -H "Content-Type: application/json" | python3 -m json.tool
else
  echo "Setting mock-vllm load level to ${LEVEL} ..."
  curl -sf -X POST "${NODEPORT_URL}/admin/set-load" \
    -H "Content-Type: application/json" \
    -d "{\"level\": ${LEVEL}}" | python3 -m json.tool
fi
