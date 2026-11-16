#!/usr/bin/env bash
# M1 cross-platform experiment in a throw-away Linux container (authorised 2026-11-11).
#   bash src/price_hgbc/run_xplat_docker.sh arm64|amd64
# - image python:3.11.13-slim-bookworm (the group's recorded Python 3.11.13), bundled pip (no upgrade)
# - installs exactly the pinned lines of the kit's requirements.txt; transitive deps constrained to the
#   macOS env's freeze (results/price_hgbc/selfcheck/pip_freeze_after.txt) so nothing floats
#   (matplotlib/notebook are unpinned and notebook-only: not installed)
# - project mounted read-only; only results/price_hgbc/selfcheck is writable
# - two repetitions + one fingerprint; the container is removed at the end (images/volumes are kept)
set -euo pipefail
PLAT="$1"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMG="python:3.11.13-slim-bookworm"
NAME="m1x-${PLAT}"
OUT="$ROOT/results/price_hgbc/selfcheck/xplat"
mkdir -p "$OUT"
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT

docker run -d --name "$NAME" --platform "linux/${PLAT}" \
  -v "$ROOT":/work:ro \
  -v "$ROOT/results/price_hgbc/selfcheck":/work/results/price_hgbc/selfcheck:rw \
  -e PYTHONDONTWRITEBYTECODE=1 -e PIP_DISABLE_PIP_VERSION_CHECK=1 -e PIP_ROOT_USER_ACTION=ignore \
  -w /work "$IMG" sleep infinity >/dev/null
docker image inspect --format '{{.Id}} {{.Os}}/{{.Architecture}}' "$IMG" > "$OUT/image_linux-${PLAT}.txt" || true
docker exec "$NAME" sh -c "uname -a; python -VV; pip --version" | tee "$OUT/container_linux-${PLAT}.txt"

docker exec "$NAME" sh -c "grep -E '^[A-Za-z0-9_.-]+==[^ ]+' comp9417_project_kit/requirements.txt > /tmp/req.txt && cat /tmp/req.txt && \
  pip install --no-cache-dir -r /tmp/req.txt -c results/price_hgbc/selfcheck/pip_freeze_after.txt" \
  > "$OUT/pip_install_linux-${PLAT}.log" 2>&1
docker exec "$NAME" pip freeze > "$OUT/pip_freeze_linux-${PLAT}.txt"

for k in 1 2; do
  docker exec "$NAME" python -m src.price_hgbc.m1_xplat --env "linux-${PLAT}" --run "$k" 2>&1 | grep -v Warning | grep -v '^  '
done
docker exec "$NAME" python /work/src/price_hgbc/env_fingerprint.py --kit /work/comp9417_project_kit \
  --out "/work/results/price_hgbc/selfcheck/xplat/fp_linux-${PLAT}.json" 2>&1 | grep -v Warning | grep -v '^  '
echo "done ${PLAT}"
