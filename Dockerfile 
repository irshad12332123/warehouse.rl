# Dockerfile — Warehouse Robot Dispatch OpenEnv
#
# Build:  docker build -t warehouse-robot-dispatch .
# Run:    docker run -p 8000:8000 warehouse-robot-dispatch
# Test:   curl -X POST http://localhost:8000/reset \
#              -H "Content-Type: application/json" \
#              -d '{"task": "multi_pick"}'

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source — only what the env needs, no training/agent code
COPY src/env/warehouse_env.py  ./src/env/warehouse_env.py
COPY src/env/entities.py       ./src/env/entities.py
COPY src/env/grid.py           ./src/env/grid.py
COPY src/env/utils.py          ./src/env/utils.py
COPY src/__init__.py           ./src/__init__.py

# Server and config
COPY server.py      .
COPY inference.py   .
COPY openenv.yaml   .

# Ensure Python package structure
RUN touch src/env/__init__.py

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "server:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-keep-alive", "30"]
