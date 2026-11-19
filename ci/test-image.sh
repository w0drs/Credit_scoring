#!/usr/bin/env bash
set -euo pipefail
image="$1"
state=$(mktemp -d)
container=
cleanup() {
    if [[ -n "$container" ]]; then
        docker rm -f "$container" >/dev/null
    fi
    rm -rf "$state"
}
trap cleanup EXIT
docker run --rm --workdir /app/tests/unit_tests "$image" python -m pytest -q
docker run --rm --user "$(id -u):$(id -g)" -v "$state:/state" "$image" python ci/verify-database.py /state
container=$(docker run -d --user "$(id -u):$(id -g)" --workdir /app/src/backend \
    -v "$state:/state:ro" -e SCORING_DB=/state/prod.db -p 127.0.0.1::8000 \
    "$image" uvicorn api:app --host 0.0.0.0 --port 8000)
address=$(docker port "$container" 8000/tcp)
for _ in {1..30}; do
    if curl -fsS "http://$address/openapi.json" >/dev/null 2>&1; then
        python3 ci/verify-api.py "http://$address"
        exit 0
    fi
    sleep 1
done
docker logs "$container"
exit 1
