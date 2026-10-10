FROM python:3.13-slim

LABEL org.opencontainers.image.authors="晨星 (CJX0712)"
LABEL org.opencontainers.image.description="KGForge — knowledge-graph embedding & link prediction (KGFuse)"

# Deterministic BLAS: single-threaded so benchmark numbers are reproducible
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY pyproject.toml README.md ./
COPY kgforge ./kgforge
COPY examples ./examples
COPY tests ./tests

# HARD lint gate inside the image build (no `|| true`)
RUN pip install --no-cache-dir ruff==0.16.10 && ruff check . && ruff format --check .

CMD ["python", "examples/run_demo.py"]
