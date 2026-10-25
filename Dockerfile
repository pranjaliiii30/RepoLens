FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 REPOLENS_DATA_DIR=/data
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-optional.txt ./
RUN pip install -r requirements.txt && (pip install -r requirements-optional.txt || true)
COPY . .
RUN useradd -m repolens && mkdir -p /data && chown -R repolens /data /app
USER repolens
VOLUME /data
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
