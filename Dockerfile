FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    libssl-dev \
    libffi-dev \
    python3-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install wheel setuptools && \
    pip install -r requirements.txt

COPY . .

EXPOSE 8000 8501