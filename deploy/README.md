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
  --label traefik.enable=true \
  --label 'traefik.http.routers.ocr-playground-api.rule=Host(`ocr-playground.srv1567269.hstgr.cloud`) && PathPrefix(`/api`)' \
  --label traefik.http.routers.ocr-playground-api.entrypoints=websecure \
  --label traefik.http.routers.ocr-playground-api.tls=true \
  --label traefik.http.routers.ocr-playground-api.tls.certresolver=letsencrypt \
  --label traefik.http.services.ocr-playground-api.loadbalancer.server.port=8090 \
  ocr-playground-api:latest
```

It is unauthenticated and public. The only defences are the caps in `api/server.py` (20 MB, 5 pages,
120 s, two parses at a time) and the fact that nothing it receives is written to disk outside a
temporary directory. Add rate limiting before pointing it at anything that matters.

## Anywhere else

```bash
rsync -av --delete index.html assets/ user@host:/var/www/ocr-playground/
```

`deploy/nginx.conf` is a standalone server block for a plain nginx host, including the cache headers
that keep a rebuilt page from being served stale.
