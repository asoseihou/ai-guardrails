# syntax=docker/dockerfile:1
#
# Base image pinned by tag AND by the digest of its multi-arch index, so a re-tagged
# image can never slip in. Update both together (Dependabot proposes the bump):
#   docker buildx imagetools inspect python:3.12.16-slim-trixie   # -> Digest: sha256:...
FROM python:3.12.16-slim-trixie@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b

# CPU-only PyTorch, exact version, from the official CPU index (installed with --no-deps:
# its pure-Python dependencies are already pinned and hashed in requirements.lock).
ARG TORCH_VERSION=2.15.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DOCAI_HOME=/app \
    DOCAI_DB_PATH=/app/var/docai.db \
    DOCAI_LOG_JSON=true

WORKDIR /app

# 1. Every runtime dependency: exact versions, hash-checked (`make lock` regenerates it).
COPY requirements.lock ./
RUN pip install --require-hashes -r requirements.lock \
 && pip install --no-deps --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"

# 2. The package itself, without resolving anything else.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps --no-build-isolation .

COPY knowledge ./knowledge
COPY web ./web
COPY samples ./samples

# 3. Train the classifier at build time on a throwaway synthetic corpus, so the
#    container starts ready and offline (no API keys, no data at runtime).
RUN python -m docai synth --n 200 --out /tmp/corpus \
 && python -m docai train --data /tmp/corpus --out /app/models/doctype_classifier.pt \
 && rm -rf /tmp/corpus

# 4. Unprivileged user. Code stays root-owned and read-only; only /app/var (SQLite for
#    the review queue and run history) is writable, so the root filesystem can be read-only.
RUN useradd --create-home --uid 10001 docai && mkdir -p /app/var && chown docai:docai /app/var
USER docai
VOLUME ["/app/var"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

# 0.0.0.0 inside the container only; docker-compose publishes it on 127.0.0.1.
CMD ["uvicorn", "docai.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header", "--no-access-log"]
