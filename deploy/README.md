# Deploying

The site is static: `index.html` plus `assets/pages/*.jpg`.

```bash
make site
rsync -av --delete index.html assets/ user@host:/var/www/ocr-playground/
```

On the server, point nginx at that directory (`deploy/nginx.conf`) or drop the files in any web root.
`records/`, `evidence.json` and the Python are only needed to *rebuild* the page, not to serve it.
