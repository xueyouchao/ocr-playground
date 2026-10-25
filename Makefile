# Everything the playground needs, in the order it needs it.
PY        ?= python3
PORT      ?= 8080

.PHONY: help pages records site serve clean

help:
	@echo "make pages    render corpus/document.pdf -> /tmp/ocr-playground/pages + assets/pages"
	@echo "make records  re-run a parser (see README); records/ is already populated"
	@echo "make site     records/ -> evidence.json -> index.html"
	@echo "make serve    serve the site on :$(PORT)"
	@echo "make clean    drop evidence.json and index.html"

pages:
	$(PY) prepare.py

site:
	$(PY) collect.py
	$(PY) build.py

serve:
	$(PY) -m http.server $(PORT) --bind 0.0.0.0

clean:
	rm -f evidence.json index.html
