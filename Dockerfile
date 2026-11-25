# Reproducible image for the dashboard, API and scheduler.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgomp is needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[postgres]"
COPY app ./app
COPY config ./config

# data (database, downloads, reports) lives in a volume
VOLUME ["/app/data"]
EXPOSE 8501 8000

HEALTHCHECK --interval=60s --timeout=5s --retries=3 CMD curl -fs http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
