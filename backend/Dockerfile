FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# git: clone/commit/push repositories.  nodejs+npm: run `npm test` / `npm run build`
# in cloned JavaScript projects.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git nodejs npm ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run as an unprivileged user; cloned repositories live in /data/workspace.
RUN useradd --create-home --shell /bin/bash app \
    && mkdir -p /data/workspace \
    && chown -R app:app /data /app \
    && chmod +x /app/docker-entrypoint.sh
USER app

ENV DEVPILOT_WORKSPACE_ROOT=/data/workspace

EXPOSE 8000
ENTRYPOINT ["/app/docker-entrypoint.sh"]
