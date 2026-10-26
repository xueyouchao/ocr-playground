# Everything the playground needs, in the order it needs it.
PY        ?= python3
PORT      ?= 8080
SITE_DIR  ?= /home/ubuntu/sites/ocr-playground
COCOINDEX ?= cocoindex

.PHONY: help pages records site serve publish api index clean

help:
	@echo "make pages    render corpus/document.pdf -> /tmp/ocr-playground/pages + assets/pages"
	@echo "make records  re-run a parser (see README); records/ is already populated"
	@echo "make site     records/ -> evidence.json -> index.html"
	@echo "make serve    serve the site on :$(PORT)"
	@echo "make publish  copy index.html + assets/ to $(SITE_DIR) (see deploy/)"
	@echo "make api      run the upload service on :8090 (GLM_OCR_BASE_URL=… for GLM-OCR)"
	@echo "make index    read OCR_CORPUS into out/ incrementally (needs requirements-cocoindex.txt)"
	@echo "make clean    drop evidence.json and index.html"

pages:
	$(PY) prepare.py

# records/ is already populated. Re-running a parser is a per-parser command with its own environment
# (see README), so this target explains itself rather than exiting 0 having done nothing.
records:
	@echo "records/ is populated; re-run one parser from runners/ or modal/ (README 'Run it'), then 'make site'"


site:
	$(PY) collect.py
	$(PY) build.py

serve:
	$(PY) -m http.server $(PORT) --bind 0.0.0.0

api:
	$(PY) api/server.py --port 8090

index:
	$(COCOINDEX) update main.py

publish: site
	# two explicit syncs: `rsync -a --delete index.html assets/ dest/` would flatten assets/
	# into the site root and then delete the real directory
	rsync -a index.html $(SITE_DIR)/
	rsync -a --delete assets/ $(SITE_DIR)/assets/

clean:
	rm -f evidence.json index.html
