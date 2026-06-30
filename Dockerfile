FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# opencv-python-headless runtime dependency
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py model.py xai.py app.py ./

EXPOSE 8080

# Shell form so $PORT (set by Cloud Run) is expanded at runtime.
CMD streamlit run app.py \
    --server.port=$PORT --server.address=0.0.0.0 \
    --server.headless=true --browser.gatherUsageStats=false \
    --server.enableCORS=false --server.enableXsrfProtection=false
