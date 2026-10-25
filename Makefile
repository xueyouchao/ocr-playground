# Everything the playground needs, in the order it needs it.
PY        ?= python3
PORT      ?= 8080
SITE_DIR  ?= /home/ubuntu/sites/ocr-playground

.PHONY: help pages records site serve publish clean

help:
	@echo "make pages    render corpus/document.pdf -> /tmp/ocr-playground/pages + assets/pages"
	@echo "make records  re-run a parser (see README); records/ is already populated"
	@echo "make site     records/ -> evidence.json -> index.html"
	@echo "make serve    serve the site on :$(PORT)"
	@echo "make publish  copy index.html + assets/ to $(SITE_DIR) (see deploy/)"
	@echo "make clean    drop evidence.json and index.html"

pages:
	$(PY) prepare.py

site:
	$(PY) collect.py
	$(PY) build.py

serve:
	$(PY) -m http.server $(PORT) --bind 0.0.0.0

publish: site
	# two explicit syncs: `rsync -a --delete index.html assets/ dest/` would flatten assets/
	# into the site root and then delete the real directory
	rsync -a index.html $(SITE_DIR)/
	rsync -a --delete assets/ $(SITE_DIR)/assets/

clean:
	rm -f evidence.json index.html
