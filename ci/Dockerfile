FROM ghcr.io/astral-sh/uv:0.13.6 AS uv
FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python -c "from pathlib import Path; p=Path('requirements.txt'); p.write_text(p.read_text(encoding='utf-16'), encoding='utf-8')" \
    && uv pip install --system --no-cache --only-binary :all: -r requirements.txt yoyo-migrations==9.0.0
COPY src ./src
COPY config ./config
COPY model ./model
COPY tests ./tests
COPY ci ./ci
COPY db ./db
COPY data/database/applications.db ./data/database/applications.db
ENV PYTHONPATH=/app
EXPOSE 8000 8501
