# The upload parser needs a rasteriser, an OCR engine and an image converter; none of them are Python.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends poppler-utils tesseract-ocr imagemagick \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir liteparse==2.17.0

WORKDIR /app
# Built from the repository root: the reading lives in pipeline.py, shared with the indexer.
COPY pipeline.py /app/pipeline.py
COPY api/server.py /app/api/server.py

EXPOSE 8090
CMD ["python", "api/server.py", "--host", "0.0.0.0", "--port", "8090"]
