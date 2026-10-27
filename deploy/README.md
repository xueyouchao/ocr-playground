# Deploying

The site is static: `index.html` plus `pages/*.jpg`. Nothing else is needed at runtime.

## This box (srv1567269.hstgr.cloud)

Traefik owns :80/:443 and terminates TLS with Let's Encrypt (HTTP-01). Sites are `nginx:alpine`
containers carrying routing labels; Traefik watches Docker and issues the certificate on first request.

```bash
make publish          # rsync site/index.html + site/pages/ into /home/ubuntu/sites/ocr-playground
```

The container behind https://ocr-playground.srv1567269.hstgr.cloud was created once:

```bash
docker run -d --name ocr-playground --restart unless-stopped \
  -v /home/ubuntu/sites/ocr-playground:/usr/share/nginx/html:ro \
  --label traefik.enable=true \
  --label 'traefik.http.routers.ocr-playground.rule=Host(`ocr-playground.srv1567269.hstgr.cloud`)' \
  --label traefik.http.routers.ocr-playground.entrypoints=websecure \
  --label traefik.http.routers.ocr-playground.tls=true \
  --label traefik.http.routers.ocr-playground.tls.certresolver=letsencrypt \
  --label traefik.http.services.ocr-playground.loadbalancer.server.port=80 \
  nginx:alpine
```

Because the directory is bind-mounted, `make publish` is the whole deploy — no restart, no reload.

The upload API is its own container, sharing the hostname under `/api` (Traefik picks the longer rule):

```bash
# the context is the repository root: the image carries src/ beside scripts/api.py
docker build -t ocr-playground-api:latest -f deploy/api.Dockerfile .
docker run -d --name ocr-playground-api --restart unless-stopped \
  -e GLM_OCR_BASE_URL=https://<workspace>--ocr-playground-glm-ocr-serve.modal.run/v1 \
  -e CHANDRA_OCR_BASE_URL=https://<workspace>--ocr-playground-chandra-web.modal.run/v1 \
  -e NEMOTRON_PARSE_BASE_URL=https://<workspace>--ocr-playground-nemotron-web.modal.run/v1 \
  -e UNLIMITED_OCR_BASE_URL=https://<workspace>--ocr-playground-unlimited-web.modal.run/v1 \
  -e MINERU_BASE_URL=https://<workspace>--ocr-playground-mineru-web.modal.run/v1 \
  -e PADDLE_OCR_BASE_URL=https://<workspace>--ocr-playground-paddle-web.modal.run/v1 \
  --label traefik.enable=true \
  --label 'traefik.http.routers.ocr-playground-api.rule=Host(`ocr-playground.srv1567269.hstgr.cloud`) && PathPrefix(`/api`)' \
  --label traefik.http.routers.ocr-playground-api.entrypoints=websecure \
  --label traefik.http.routers.ocr-playground-api.tls=true \
  --label traefik.http.routers.ocr-playground-api.tls.certresolver=letsencrypt \
  --label traefik.http.services.ocr-playground-api.loadbalancer.server.port=8090 \
  ocr-playground-api:latest
```

Each `*_BASE_URL` points at the app from the matching `modal deploy deploy/modal/serve_*.py`; Modal prints
the URL, and the served name follows the wrapper. `PADDLE_OCR_BASE_URL` comes from
`modal deploy deploy/modal/serve_paddle.py` (app `ocr-playground-paddle`, an A10G because the L4 pool was
dry). Leave any variable out and that mode is listed but greyed out rather than failing an upload;
with `GLM_OCR_BASE_URL` missing, the page selects the fast path (LiteParse + Tesseract) instead. The
GPUs scale to zero, so the first upload after idle waits minutes for the model to boot; the container
holds the request for up to `GLM_OCR_BUDGET_SECONDS` (900 s) rather than giving up.

It is unauthenticated and public, and so is the Modal endpoint behind it. The only defences are the
caps in `scripts/api.py` (20 MB, 5 pages, 120 s per fast parse, 900 s for a whole GLM document, two
parses at a time) and the fact that nothing it receives is written to disk outside a temporary
directory. Add rate limiting before pointing it at anything that matters.

## Anywhere else

```bash
rsync -av site/index.html user@host:/var/www/ocr-playground/
rsync -av --delete site/pages/ user@host:/var/www/ocr-playground/pages/
```

`deploy/nginx.conf` is a standalone server block for a plain nginx host, including the cache headers
that keep a rebuilt page from being served stale.
