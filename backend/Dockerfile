FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY backend/requirements.txt /tmp/requirements.txt
COPY requirements-test.txt /tmp/requirements-test.txt
ARG TORCH_VERSION=2.6.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --upgrade pip \
    && pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" \
    && pip install -r /tmp/requirements.txt

COPY backend /app/backend
COPY config /app/config

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
