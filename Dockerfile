# syntax=docker/dockerfile:1

FROM python:3.12-slim

# Never write .pyc, never buffer stdout — you want logs to appear immediately
# in `docker logs`, not when the buffer happens to flush.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Dependencies are copied and installed BEFORE the application code. Docker
# caches each layer, so editing app/main.py does not reinstall FastAPI —
# rebuilds drop from ~40s to ~2s. Getting this order wrong is the single most
# common Dockerfile mistake.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app


# Run as a non-root user. If the process is compromised it should not own the
# filesystem. Do this after the installs, which need root.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /srv
USER appuser

EXPOSE 8000

# The container runtime uses this to decide whether the container is alive.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

# 0.0.0.0, not 127.0.0.1. Binding to loopback inside a container means nothing
# outside the container can ever reach it — a classic first-Docker-app bug.
#
# One worker is correct here: the workload is I/O-bound and async, so a single
# event loop handles high concurrency (proven: 8 x 1s requests in 1.2s).
# Scale by running more containers, not more workers per container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
