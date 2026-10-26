# Deploying

The site is static: `index.html` plus `assets/pages/*.jpg`. Nothing else is needed at runtime.

## This box (srv1567269.hstgr.cloud)

Traefik owns :80/:443 and terminates TLS with Let's Encrypt (HTTP-01). Sites are `nginx:alpine`
containers carrying routing labels; Traefik watches Docker and issues the certificate on first request.

```bash
make publish          # rsync index.html + assets/ into /home/ubuntu/sites/ocr-playground
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
docker build -t ocr-playground-api:latest -f api/Dockerfile api
docker run -d --name ocr-playground-api --restart unless-stopped \
  -e GLM_OCR_BASE_URL=https://<workspace>--ocr-playground-glm-ocr-serve.modal.run/v1 \
  --label traefik.enable=true \
  --label 'traefik.http.routers.ocr-playground-api.rule=Host(`ocr-playground.srv1567269.hstgr.cloud`) && PathPrefix(`/api`)' \
  --label traefik.http.routers.ocr-playground-api.entrypoints=websecure \
  --label traefik.http.routers.ocr-playground-api.tls=true \
  --label traefik.http.routers.ocr-playground-api.tls.certresolver=letsencrypt \
  --label traefik.http.services.ocr-playground-api.loadbalancer.server.port=8090 \
  ocr-playground-api:latest
```

`GLM_OCR_BASE_URL` points at the app from `modal deploy modal/serve_glm_ocr.py`; Modal prints the URL,
and the served name is `glm-ocr`. Leave the variable out and the container still serves, with GLM-OCR
reported unavailable and the fast path (LiteParse + Tesseract) the only one — the page will say so and
select it. The GPU scales to zero, so the first upload after idle waits minutes for vLLM to boot; the
container holds the request for up to `GLM_OCR_BUDGET_SECONDS` (900 s) rather than giving up.

It is unauthenticated and public, and so is the Modal endpoint behind it. The only defences are the
caps in `api/server.py` (20 MB, 5 pages, 120 s per fast parse, 900 s for a whole GLM document, two
parses at a time) and the fact that nothing it receives is written to disk outside a temporary
directory. Add rate limiting before pointing it at anything that matters.

## Anywhere else

```bash
rsync -av --delete index.html assets user@host:/var/www/ocr-playground/
```

`deploy/nginx.conf` is a standalone server block for a plain nginx host, including the cache headers
that keep a rebuilt page from being served stale.
