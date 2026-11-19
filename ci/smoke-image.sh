#!/usr/bin/env bash
set -euo pipefail
image="$1"
database="$2"
stand="$3"
container=$(docker run -d --user "$(id -u):$(id -g)" --workdir /app/src/backend \
    -v "$database:/state/scoring.db:ro" -e SCORING_DB=/state/scoring.db -e OMP_NUM_THREADS=2 \
    -p 127.0.0.1::8000 "$image" uvicorn api:app --host 0.0.0.0 --port 8000)
trap 'docker rm -f "$container" >/dev/null' EXIT
address=$(docker port "$container" 8000/tcp)
for _ in {1..30}; do
    if curl -fsS "http://$address/openapi.json" >/dev/null 2>&1; then
        echo "Deployed $stand container ${container:0:12} at $address"
        python3 ci/verify-api.py "http://$address"
        exit 0
    fi
    sleep 1
done
docker logs "$container"
exit 1
