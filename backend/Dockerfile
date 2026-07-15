# syntax=docker/dockerfile:1
FROM python:3.12-slim

# System deps for unstructured / presidio / builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# --retries and --timeout ride out transient PyPI hiccups on the big wheels
# (torch, spaCy models). The pip cache mount means that if a download drops
# mid-stream, retrying the build doesn't have to redo packages that already
# came down fine. --force-reinstall clears out any partially-downloaded wheel
# BuildKit's cache left behind from a failed prior build (that's what was
# causing the "does not match the hashes" error).
#
# sentence-transformers pulls in torch transitively, and PyPI's default torch
# wheel bundles full CUDA support (~500MB+) we don't need since embeddings and
# reranking run on CPU here. That's large enough that one dropped connection
# wastes the whole download, so we grab the much smaller CPU-only wheel from
# PyTorch's own index first.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install \
        --retries 10 \
        --timeout 120 \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install \
        --force-reinstall \
        --retries 10 \
        --timeout 120 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Presidio spaCy model
RUN python -m spacy download en_core_web_sm

COPY . .

RUN mkdir -p /app/uploads

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
