# Dockerfile — OpenEnv Warehouse Submission (Final)

FROM python:3.11-slim

# ---- Environment ----
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ---- System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---- Working dir ----
WORKDIR /app

# ---- Install Python deps (cached layer) ----
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefer-binary --timeout 1000 --retries 10 -r requirements.txt

# ---- Copy project files ----
COPY src/ ./src/
COPY server.py .
COPY inference.py .
COPY openenv.yaml .

# Ensure package structure (safe)
RUN touch src/env/__init__.py

# ---- Expose API port ----
EXPOSE 8000

# ---- Start server (HF requirement: bind 0.0.0.0) ----
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
