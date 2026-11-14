#!/usr/bin/env bash
# Dump the full tuned_C model inside a throw-away Linux container (same image / pins / mounts as
# run_xplat_docker.sh).   bash src/price_hgbc/run_xplat_trees_docker.sh arm64|amd64
set -euo pipefail
PLAT="$1"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMG="python:3.11.13-slim-bookworm"
NAME="m1x-trees-${PLAT}"
OUT="$ROOT/results/price_hgbc/selfcheck/xplat"
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT
docker run -d --name "$NAME" --platform "linux/${PLAT}" \
  -v "$ROOT":/work:ro \
  -v "$ROOT/results/price_hgbc/selfcheck":/work/results/price_hgbc/selfcheck:rw \
  -e PYTHONDONTWRITEBYTECODE=1 -e PIP_DISABLE_PIP_VERSION_CHECK=1 -e PIP_ROOT_USER_ACTION=ignore \
  -w /work "$IMG" sleep infinity >/dev/null
docker exec "$NAME" sh -c "grep -E '^[A-Za-z0-9_.-]+==[^ ]+' comp9417_project_kit/requirements.txt > /tmp/req.txt && \
  pip install --no-cache-dir -r /tmp/req.txt -c results/price_hgbc/selfcheck/pip_freeze_after.txt" \
  > "$OUT/pip_install_trees_linux-${PLAT}.log" 2>&1
docker exec "$NAME" pip freeze > "$OUT/pip_freeze_trees_linux-${PLAT}.txt"
docker exec "$NAME" python -m src.price_hgbc.m1_xplat_trees --env "linux-${PLAT}" 2>&1 | grep -v Warning | grep -v '^  '
echo "done trees ${PLAT}"
