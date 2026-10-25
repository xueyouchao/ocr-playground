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

## Anywhere else

```bash
rsync -av --delete index.html assets/ user@host:/var/www/ocr-playground/
```

`deploy/nginx.conf` is a standalone server block for a plain nginx host, including the cache headers
that keep a rebuilt page from being served stale.
