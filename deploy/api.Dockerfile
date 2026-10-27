# The upload parser needs a rasteriser, an OCR engine and an image converter; none of them are Python.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends poppler-utils tesseract-ocr imagemagick \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Built from the repository root. The service is the package; scripts/api.py is the compatibility
# entry point the CMD still uses. dependency-injector is pinned in requirements.txt.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt liteparse==2.17.0

COPY src /app/src
ENV PYTHONPATH=/app/src
COPY scripts/api.py /app/scripts/api.py

EXPOSE 8090
CMD ["python", "scripts/api.py", "--host", "0.0.0.0", "--port", "8090"]
