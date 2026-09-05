FROM python:3.11-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR $HOME/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt && pip check
COPY --chown=user . .
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:7860/healthz', timeout=3).read()" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
