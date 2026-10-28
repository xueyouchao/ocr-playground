#!/usr/bin/env bash
# Build and (re)start the upload API container on this host.
#
# The values here are the ones the running container was started with — they used to live in a scratch
# script in /tmp, which meant the deployment existed only on the machine that ran it. Override any of
# them from the environment to point at a different workspace or host:
#
#   API_HOST=… GLM_OCR_BASE_URL=… deploy/run_api.sh
#
# Recreating the container kills any parse in flight, and a Modal cold start keeps going server-side
# after its client is gone, so a boot you cannot see in the socket table may still be running. Check
# `modal container list` before deploying while somebody is testing.
set -euo pipefail

cd "$(dirname "$0")/.."

API_HOST=${API_HOST:-ocr-playground.srv1567269.hstgr.cloud}
WORKSPACE=${WORKSPACE:-xueyouchao}
MODAL_HOST=${MODAL_HOST:-us-east.modal.run}
IMAGE=${IMAGE:-ocr-playground-api:latest}

url() { echo "https://${WORKSPACE}--ocr-playground-$1-web.${MODAL_HOST}/v1"; }

# `GLM_OCR_WARM_DAILY_CAP` is deliberately not set: the code default (100 per engine per rolling day)
# is the number the docs describe, and overriding it here is how the deployed cap drifted to 20.
[[ ${BUILD:-1} == 0 ]] || docker build -t "$IMAGE" -f deploy/api.Dockerfile .

docker rm -f ocr-playground-api >/dev/null 2>&1 || true
docker run -d --name ocr-playground-api --restart unless-stopped \
  -e "CHANDRA_OCR_BASE_URL=$(url chandra)" \
  -e "GLM_OCR_BASE_URL=https://${WORKSPACE}--ocr-playground-glm-ocr-serve.${MODAL_HOST}/v1" \
  -e "MINERU_BASE_URL=$(url mineru)" \
  -e "NEMOTRON_PARSE_BASE_URL=$(url nemotron)" \
  -e "PADDLE_OCR_BASE_URL=$(url paddle)" \
  -e "UNLIMITED_OCR_BASE_URL=$(url unlimited)" \
  -e PYTHONPATH=/app/src \
  --label traefik.enable=true \
  --label "traefik.http.routers.ocr-playground-api.rule=Host(\`${API_HOST}\`) && PathPrefix(\`/api\`)" \
  --label traefik.http.routers.ocr-playground-api.entrypoints=websecure \
  --label traefik.http.routers.ocr-playground-api.tls=true \
  --label traefik.http.routers.ocr-playground-api.tls.certresolver=letsencrypt \
  --label traefik.http.services.ocr-playground-api.loadbalancer.server.port=8090 \
  "$IMAGE"

# The health line is part of the script: a container that starts and cannot resolve an engine is up,
# and looks fine until somebody uploads.
for _ in $(seq 1 20); do
  if curl -fsS -o /dev/null "https://${API_HOST}/api/health"; then
    echo "api up: https://${API_HOST}/api/health"
    exit 0
  fi
  sleep 1
done
echo "the container is running but /api/health did not answer; check 'docker logs ocr-playground-api'" >&2
exit 1
