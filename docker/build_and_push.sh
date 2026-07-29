#!/usr/bin/env bash
# Build the diffusion-boundary Blackwell image and push to GHCR.
#
# Prereqs (you do these once):
#   1. Create a GitHub PAT (classic) with scope: write:packages
#   2. docker login ghcr.io -u <your-github-user>   (paste PAT as password)
#
# Then run from the docker/ directory:
#   GHCR_USER=<your-github-user> bash build_and_push.sh
#
# The HF token is read from ../.env and passed as a BuildKit *secret*
# (needed for the gated SD v1-4 download); it never lands in an image layer.
set -euo pipefail

GHCR_USER="${GHCR_USER:?set GHCR_USER=<your-github-username>}"
TAG="${TAG:-cu128}"
IMAGE="ghcr.io/${GHCR_USER}/diffusion-boundary:${TAG}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# --- extract HF token from repo .env into a temp secret file (never printed) ---
ENV_FILE="$HERE/../.env"
[ -f "$ENV_FILE" ] || { echo "ERROR: $ENV_FILE not found (need HF_TOKEN)"; exit 1; }
SECRET_FILE="$(mktemp)"
trap 'rm -f "$SECRET_FILE"' EXIT
grep '^HF_TOKEN=' "$ENV_FILE" | head -1 | cut -d= -f2- > "$SECRET_FILE"
[ -s "$SECRET_FILE" ] || { echo "ERROR: HF_TOKEN empty in .env"; exit 1; }

echo "==> building $IMAGE (this pulls deps + torch cu128 + ~7GB model weights)"
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,src="$SECRET_FILE" \
  -t "$IMAGE" \
  -f "$HERE/Dockerfile" \
  "$HERE"

echo "==> pushing $IMAGE"
docker push "$IMAGE"

echo "DONE: $IMAGE"
echo "Set the package visibility (Private recommended) at:"
echo "  https://github.com/users/${GHCR_USER}/packages/container/diffusion-boundary/settings"
